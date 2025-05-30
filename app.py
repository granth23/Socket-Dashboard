from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.signal import find_peaks
import socket
import struct
import threading
import time
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vibration_analysis_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Constants - must match Arduino
BUFFER_SAMPLES = 131072
TOTAL_FLOATS = BUFFER_SAMPLES * 3
BYTES_NEEDED = TOTAL_FLOATS * 4

G = 9.80665
TO_MMPS = 1000

# Global variables for sharing data between threads
latest_data = {
    'accel_time': {'x': [], 'y': [], 'z': [], 'rms': [0, 0, 0]},
    'accel_fft': {'freqs': [], 'x': [], 'y': [], 'z': [], 'peaks': {}},
    'vel_time': {'x': [], 'y': [], 'z': [], 'rms': [0, 0, 0]},
    'vel_fft': {'freqs': [], 'x': [], 'y': [], 'z': [], 'peaks': {}}
}

# ESP32 connection status
esp32_status = {
    'connected': False,
    'last_data_time': 0,
    'client_address': None
}

def accel_fft(a, N, dt):
    return np.fft.fftfreq(N, d=dt), np.fft.fft(a)

def freq_integration(freqs, A_fft, N):
    eps = 1e-12
    denom = 2j * np.pi * freqs
    denom[0] = eps
    V_fft = A_fft / denom
    V_fft[0] = 0
    v_time = np.fft.ifft(V_fft).real
    mask = freqs > 0
    mag = (2.0/N) * np.abs(V_fft[mask])
    return v_time, freqs[mask], mag

def find_peak_annotation(freqs, mags):
    peaks, _ = find_peaks(mags)
    if not len(peaks):
        return None
    idx = peaks[np.argmax(mags[peaks])]
    f, m = freqs[idx], mags[idx]
    return {'freq': float(f), 'mag': float(m)}

def calculate_peak_rms(freqs, mags, peaks):
    if not peaks:
        return 0
    peak_indices = [np.argmin(np.abs(freqs - p['freq'])) for p in peaks if p is not None]
    peak_mags = [mags[i] for i in peak_indices]
    return np.sqrt(np.mean(np.square(peak_mags)))

def tcp_receiver():
    """TCP server thread to receive data from ESP32"""
    global latest_data, esp32_status
    
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', 12345))
    srv.listen(1)
    print("Waiting for ESP32 on port 12345...")
    
    while True:
        try:
            conn, addr = srv.accept()
            print(f"ESP32 connected from {addr}")
            esp32_status['connected'] = True
            esp32_status['client_address'] = addr[0]
            
            # Notify all web clients about ESP32 connection
            socketio.emit('esp32_status', esp32_status)
            
            while True:
                # Receive exactly BUFFER_SAMPLES×3 floats
                buf = b''
                while len(buf) < BYTES_NEEDED:
                    chunk = conn.recv(4096)
                    if not chunk:
                        raise RuntimeError("Socket closed")
                    buf += chunk

                # Unpack & reshape
                data = np.array(struct.unpack('<'+'f'*TOTAL_FLOATS, buf),
                              dtype=np.float32).reshape(-1, 3)

                # Scale & remove DC offset
                arr_x = data[:, 0] * G
                arr_x -= arr_x.mean()
                arr_y = data[:, 1] * G
                arr_y -= arr_y.mean()
                arr_z = data[:, 2] * G
                arr_z -= arr_z.mean()

                N = BUFFER_SAMPLES
                fs = 26667
                dt = 1/fs

                # FFT acceleration
                fx, Ax = accel_fft(arr_x, N, dt)
                _, Ay = accel_fft(arr_y, N, dt)
                _, Az = accel_fft(arr_z, N, dt)

                # Frequency-domain integration → velocity
                vel_x, vfx, Vx = freq_integration(fx, Ax, N)
                vel_y, _, Vy = freq_integration(fx, Ay, N)
                vel_z, _, Vz = freq_integration(fx, Az, N)

                # Convert to mm/s
                vel_x *= TO_MMPS
                vel_y *= TO_MMPS
                vel_z *= TO_MMPS
                Vx *= TO_MMPS
                Vy *= TO_MMPS
                Vz *= TO_MMPS

                # RMS values
                rms_acc = [np.sqrt(np.mean(arr_x**2)),
                          np.sqrt(np.mean(arr_y**2)),
                          np.sqrt(np.mean(arr_z**2))]
                rms_vel = [np.sqrt(np.mean(vel_x**2)),
                          np.sqrt(np.mean(vel_y**2)),
                          np.sqrt(np.mean(vel_z**2))]

                # Prepare frequency domain data (positive frequencies only)
                mask = fx > 0
                fxp = fx[mask]
                Ax_mag = (2.0/N) * np.abs(Ax[mask])
                Ay_mag = (2.0/N) * np.abs(Ay[mask])
                Az_mag = (2.0/N) * np.abs(Az[mask])

                # Find peaks
                accel_peaks = {
                    'X': find_peak_annotation(fxp, Ax_mag),
                    'Y': find_peak_annotation(fxp, Ay_mag),
                    'Z': find_peak_annotation(fxp, Az_mag)
                }
                vel_peaks = {
                    'X': find_peak_annotation(vfx, Vx),
                    'Y': find_peak_annotation(vfx, Vy),
                    'Z': find_peak_annotation(vfx, Vz)
                }

                # Calculate peak RMS values
                accel_peak_rms = {
                    'X': calculate_peak_rms(fxp, Ax_mag, [accel_peaks['X']]),
                    'Y': calculate_peak_rms(fxp, Ay_mag, [accel_peaks['Y']]),
                    'Z': calculate_peak_rms(fxp, Az_mag, [accel_peaks['Z']])
                }
                vel_peak_rms = {
                    'X': calculate_peak_rms(vfx, Vx, [vel_peaks['X']]),
                    'Y': calculate_peak_rms(vfx, Vy, [vel_peaks['Y']]),
                    'Z': calculate_peak_rms(vfx, Vz, [vel_peaks['Z']])
                }

                # Update global data
                latest_data = {
                    'accel_time': {
                        'x': arr_x.tolist(),
                        'y': arr_y.tolist(),
                        'z': arr_z.tolist(),
                        'rms': [float(r) for r in rms_acc]
                    },
                    'accel_fft': {
                        'freqs': fxp.tolist(),
                        'x': Ax_mag.tolist(),
                        'y': Ay_mag.tolist(),
                        'z': Az_mag.tolist(),
                        'peaks': accel_peaks,
                        'peak_rms': accel_peak_rms
                    },
                    'vel_time': {
                        'x': vel_x.tolist(),
                        'y': vel_y.tolist(),
                        'z': vel_z.tolist(),
                        'rms': [float(r) for r in rms_vel]
                    },
                    'vel_fft': {
                        'freqs': vfx.tolist(),
                        'x': Vx.tolist(),
                        'y': Vy.tolist(),
                        'z': Vz.tolist(),
                        'peaks': vel_peaks,
                        'peak_rms': vel_peak_rms
                    }
                }
                # Update timestamp
                esp32_status['last_data_time'] = time.time()

                # Emit data to all connected clients
                socketio.emit('data_update', latest_data)
                
                time.sleep(0.05)
                
        except Exception as e:
            print(f"ESP32 connection error: {e}")
            esp32_status['connected'] = False
            esp32_status['client_address'] = None
            socketio.emit('esp32_status', esp32_status)
            time.sleep(1)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/api/data')
def get_data():
    return jsonify({**latest_data, 'esp32_status': esp32_status})

@app.route('/api/status')
def get_status():
    return jsonify(esp32_status)

@socketio.on('connect')
def handle_connect():
    print('Web client connected')
    emit('esp32_status', esp32_status)
    if latest_data['accel_time']['x']:  # Only send data if we have some
        emit('data_update', latest_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Web client disconnected')

if __name__ == '__main__':
    # Start TCP receiver in background thread
    tcp_thread = threading.Thread(target=tcp_receiver, daemon=True)
    tcp_thread.start()
    
    # Start Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)