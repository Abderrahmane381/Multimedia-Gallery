
import requests

IMAGE_PATH = "uploads/5703572ec9214714be5742b5242b27d8.png"

endpoints = [
    "/api/color",
    "/api/texture",
    "/api/features"
]

for endpoint in endpoints:
    print("\nTEST :", endpoint)

    with open(IMAGE_PATH, "rb") as image:
        response = requests.post(
            "http://127.0.0.1:5000" + endpoint,
            files={"image": image},
            data={"mode": "global"}
        )

    print("Status :", response.status_code)

    try:
        result = response.json()
        print("Success :", result.get("success"))
        if response.status_code != 200:
            print("Erreur :", result)
    except ValueError:
        print("Réponse non JSON :", response.text[:500])
