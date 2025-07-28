from pygrabber.dshow_graph import FilterGraph
import cv2

graph = FilterGraph()
devices = graph.get_input_devices()

print("=== 使用可能なカメラ一覧 ===")
for index, name in enumerate(devices):
    print(f"Index {index}: {name}")

print("\n=== 実際に使えるカメラのみをチェック中 ===")
for index, name in enumerate(devices):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"✅ Index {index} ({name}) is available.")
        cap.release()
    else:
        print(f"❌ Index {index} ({name}) is NOT available.")