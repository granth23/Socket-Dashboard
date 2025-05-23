from flask import Flask, Response, jsonify, render_template
import threading
import time
import random
import json

app = Flask(__name__)

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

machine_data = {m_id: [] for m_id in all_machine_ids}
machine_counters = {m_id: 0 for m_id in all_machine_ids}

MAX_POINTS = 100       
DISPLAY_POINTS = 20    
VARIANCE_WINDOW = 5    
VARIANCE_THRESHOLD = 500 

def compute_variance(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return var


def background_data_generator():
    while True:
        for m_id in machine_data:
            machine_counters[m_id] += 1
            new_index = machine_counters[m_id]
            new_value = random.randint(0, 100)
            machine_data[m_id].append((new_index, new_value))
            
            if len(machine_data[m_id]) > MAX_POINTS:
                machine_data[m_id].pop(0)
        time.sleep(1)

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
            data_list = machine_data.get(machine_id, [])
            displayed_data = data_list[-DISPLAY_POINTS:]
            
            payload = {
                "machine_id": machine_id,
                "data": displayed_data,  
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1)

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
                    
                    points = machine_data[m_id][-VARIANCE_WINDOW:]
                    
                    if len(points) < VARIANCE_WINDOW:
                        continue
                    
                    values = [pt[1] for pt in points]
                    var = compute_variance(values)
                    if var > VARIANCE_THRESHOLD:
                        total_exceed += 1
                area_counts[str(area_id)] = total_exceed
            
            
            yield f"data: {json.dumps(area_counts)}\n\n"
            time.sleep(1)

    return Response(event_stream_areas(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
