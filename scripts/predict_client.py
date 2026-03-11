import requests

url = "http://127.0.0.1:8000/predict"
file_path = "data/raw/heart_test.csv" # проверьте название вашего файла

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    print("Успех! Получено предсказаний:", len(data["results"]))
    print("Пример ответа:", data["results"][:3])
else:
    print("Ошибка:", response.text)