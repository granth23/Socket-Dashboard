from flask import Flask, Response, jsonify, render_template
import threading
import time
import random
import json
import numpy as np
from scipy.signal import find_peaks
import socket
import struct

app = Flask(__name__)

# Constants matching the original vibration analysis
BUFFER_SAMPLES = 2048
G = 9.80665
TO_MMPS = 1000
SAMPLING_RATE = 26667

# ESP32 connection constants
TOTAL_FLOATS = BUFFER_SAMPLES * 3
BYTES_NEEDED = TOTAL_FLOATS * 4

factory_areas = []
for i in range(1, 11):
    factory_areas.append({
        "id": i,
        "name": f"Factory Area {i}",
        "machines": []
    })
    for j in range(1, 1000):
        factory_areas[i-1]["machines"].append({
            "id": i*1000 + j,
            "name": f"Machine {j}"
        })

all_machine_ids = [m["id"] for area in factory_areas for m in area["machines"]]

# Store vibration data for each machine
machine_vibration_data = {m_id: {
    'accel_x': [], 'accel_y': [], 'accel_z': [],
    'vel_x': [], 'vel_y': [], 'vel_z': [],
    'accel_fft_x': [], 'accel_fft_y': [], 'accel_fft_z': [],
    'vel_fft_x': [], 'vel_fft_y': [], 'vel_fft_z': [],
    'frequencies': [],
    'rms_acc': [0, 0, 0],
    'rms_vel': [0, 0, 0],
    'peaks_acc': {},
    'peaks_vel': {}
} for m_id in all_machine_ids}

machine_counters = {m_id: 0 for m_id in all_machine_ids}

MAX_POINTS = 100
DISPLAY_POINTS = 50
VARIANCE_WINDOW = 5
VARIANCE_THRESHOLD = 500

def accel_fft(a, N, dt):
    """Compute FFT of acceleration data"""
    return np.fft.fftfreq(N, d=dt), np.fft.fft(a)

def freq_integration(freqs, A_fft, N):
    """Convert acceleration FFT to velocity via frequency domain integration"""
    eps = 1e-12
    denom = 2j * np.pi * freqs
    denom[0] = eps
    V_fft = A_fft / denom
    v_time = np.fft.ifft(V_fft).real
    mask = freqs > 0
    mag = (2.0/N) * np.abs(V_fft[mask])
    return v_time, freqs[mask], mag

def find_dominant_peak(freqs, mags):
    """Find the dominant frequency peak"""
    if len(mags) == 0:
        return None
    peaks, _ = find_peaks(mags)
    if len(peaks) == 0:
        return None
    idx = peaks[np.argmax(mags[peaks])]
    return {"freq": float(freqs[idx]), "mag": float(mags[idx])}

def generate_vibration_data():
    """Generate realistic vibration data with multiple frequency components - RANDOM DATA VERSION"""
    # Generate base frequencies (simulating machine harmonics)
    base_freq = random.uniform(10, 100)  # Main machine frequency
    harmonics = [base_freq, base_freq * 2, base_freq * 3]
    
    # Generate time series
    dt = 1/SAMPLING_RATE
    t = np.linspace(0, BUFFER_SAMPLES * dt, BUFFER_SAMPLES)
    
    # Create 3-axis acceleration with multiple frequency components
    accel_x = np.zeros(BUFFER_SAMPLES)
    accel_y = np.zeros(BUFFER_SAMPLES)
    accel_z = np.zeros(BUFFER_SAMPLES)
    
    for freq in harmonics:
        amplitude_x = random.uniform(0.1, 2.0)
        amplitude_y = random.uniform(0.1, 2.0)
        amplitude_z = random.uniform(0.1, 2.0)
        phase_x = random.uniform(0, 2*np.pi)
        phase_y = random.uniform(0, 2*np.pi)
        phase_z = random.uniform(0, 2*np.pi)
        
        accel_x += amplitude_x * np.sin(2 * np.pi * freq * t + phase_x)
        accel_y += amplitude_y * np.sin(2 * np.pi * freq * t + phase_y)
        accel_z += amplitude_z * np.sin(2 * np.pi * freq * t + phase_z)
    
    # Add noise
    noise_level = 0.1
    accel_x += np.random.normal(0, noise_level, BUFFER_SAMPLES)
    accel_y += np.random.normal(0, noise_level, BUFFER_SAMPLES)
    accel_z += np.random.normal(0, noise_level, BUFFER_SAMPLES)
    
    # Scale by gravity and remove DC offset
    accel_x *= G
    accel_y *= G
    accel_z *= G
    accel_x -= accel_x.mean()
    accel_y -= accel_y.mean()
    accel_z -= accel_z.mean()
    
    return accel_x, accel_y, accel_z

# ESP32 connection variable (global)
esp32_connection = None

# def generate_vibration_data():
#     """Read vibration data from ESP32 - ESP32 DATA VERSION"""
#     global esp32_connection
    
#     try:
#         # If no connection, try to establish one
#         if esp32_connection is None:
#             srv = socket.socket()
#             srv.bind(('0.0.0.0', 12345))
#             srv.listen(1)
#             print("Waiting for ESP32 connection...")
#             esp32_connection, _ = srv.accept()
#             print("ESP32 Connected!")
        
#         # Receive exactly BUFFER_SAMPLES×3 floats
#         buf = b''
#         while len(buf) < BYTES_NEEDED:
#             chunk = esp32_connection.recv(4096)
#             if not chunk:
#                 raise RuntimeError("ESP32 socket closed")
#             buf += chunk
        
#         # Unpack & reshape data
#         data = np.array(struct.unpack('<'+'f'*TOTAL_FLOATS, buf),
#                        dtype=np.float32).reshape(-1, 3)
        
#         # Extract 3-axis acceleration data
#         accel_x = data[:, 0]
#         accel_y = data[:, 1] 
#         accel_z = data[:, 2]
        
#         # Scale by gravity and remove DC offset
#         accel_x *= G
#         accel_y *= G
#         accel_z *= G
#         accel_x -= accel_x.mean()
#         accel_y -= accel_y.mean()
#         accel_z -= accel_z.mean()
        
#         return accel_x, accel_y, accel_z
        
#     except Exception as e:
#         print(f"ESP32 connection error: {e}")
#         # Reset connection on error
#         esp32_connection = None
#         # Fall back to random data or raise error
#         raise RuntimeError("Failed to read from ESP32")

def process_vibration_data(accel_x, accel_y, accel_z):
    """Process vibration data similar to the original analysis"""
    N = BUFFER_SAMPLES
    dt = 1/SAMPLING_RATE
    
    # FFT acceleration
    fx, Ax = accel_fft(accel_x, N, dt)
    _, Ay = accel_fft(accel_y, N, dt)
    _, Az = accel_fft(accel_z, N, dt)
    
    # Frequency domain integration to get velocity
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
    
    # Calculate RMS values
    rms_acc = [
        float(np.sqrt(np.mean(accel_x**2))),
        float(np.sqrt(np.mean(accel_y**2))),
        float(np.sqrt(np.mean(accel_z**2)))
    ]
    rms_vel = [
        float(np.sqrt(np.mean(vel_x**2))),
        float(np.sqrt(np.mean(vel_y**2))),
        float(np.sqrt(np.mean(vel_z**2)))
    ]
    
    # Get positive frequencies for FFT plots
    mask = fx > 0
    fxp = fx[mask]
    Ax_mag = (2.0/N) * np.abs(Ax[mask])
    Ay_mag = (2.0/N) * np.abs(Ay[mask])
    Az_mag = (2.0/N) * np.abs(Az[mask])
    
    # Find dominant peaks
    peaks_acc = {
        'X': find_dominant_peak(fxp, Ax_mag),
        'Y': find_dominant_peak(fxp, Ay_mag),
        'Z': find_dominant_peak(fxp, Az_mag)
    }
    peaks_vel = {
        'X': find_dominant_peak(vfx, Vx),
        'Y': find_dominant_peak(vfx, Vy),
        'Z': find_dominant_peak(vfx, Vz)
    }
    
    return {
        'accel_time': {
            'x': accel_x.tolist(),
            'y': accel_y.tolist(),
            'z': accel_z.tolist()
        },
        'vel_time': {
            'x': vel_x.tolist(),
            'y': vel_y.tolist(),
            'z': vel_z.tolist()
        },
        'accel_fft': {
            'frequencies': fxp.tolist(),
            'x': Ax_mag.tolist(),
            'y': Ay_mag.tolist(),
            'z': Az_mag.tolist()
        },
        'vel_fft': {
            'frequencies': vfx.tolist(),
            'x': Vx.tolist(),
            'y': Vy.tolist(),
            'z': Vz.tolist()
        },
        'rms_acc': rms_acc,
        'rms_vel': rms_vel,
        'peaks_acc': peaks_acc,
        'peaks_vel': peaks_vel
    }

def compute_variance(values):
    """Compute variance for anomaly detection"""
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return var

def background_data_generator():
    """Generate vibration data for all machines"""
    while True:
        try:
            for m_id in machine_vibration_data:
                # Generate new vibration data
                accel_x, accel_y, accel_z = generate_vibration_data()
                processed_data = process_vibration_data(accel_x, accel_y, accel_z)
                
                # Store the processed data
                machine_vibration_data[m_id] = processed_data
                machine_counters[m_id] += 1
                
        except Exception as e:
            print(f"Data generation error: {e}")
            # Continue with next iteration on error
            
        time.sleep(2)  # Update every 2 seconds

threading.Thread(target=background_data_generator, daemon=True).start()

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/api/factory_areas', methods=['GET'])
def get_factory_areas():
    return jsonify(factory_areas)

@app.route('/stream/<int:machine_id>', methods=['GET'])
def stream_machine_data(machine_id):
    def event_stream():
        while True:
            if machine_id in machine_vibration_data:
                data = machine_vibration_data[machine_id]
                payload = {
                    "machine_id": machine_id,
                    "data": data,
                    "timestamp": time.time()
                }
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(2)
    
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/stream_areas', methods=['GET'])
def stream_areas():
    def event_stream_areas():
        while True:
            area_counts = {}
            for area in factory_areas:
                area_id = area["id"]
                total_exceed = 0
                for m in area["machines"]:
                    m_id = m["id"]
                    
                    # Use RMS acceleration as anomaly indicator
                    if m_id in machine_vibration_data:
                        rms_acc = machine_vibration_data[m_id].get('rms_acc', [0, 0, 0])
                        max_rms = max(rms_acc) if rms_acc else 0
                        if max_rms > 5.0:  # Threshold for high vibration
                            total_exceed += 1
                
                area_counts[str(area_id)] = total_exceed
            
            yield f"data: {json.dumps(area_counts)}\n\n"
            time.sleep(2)
    
    return Response(event_stream_areas(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)