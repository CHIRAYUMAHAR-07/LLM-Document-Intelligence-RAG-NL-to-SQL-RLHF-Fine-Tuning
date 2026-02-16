
from locust import HttpUser, task, between

class AIPlatformUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def test_home(self):
        self.client.get("/")

    @task
    def test_predict(self):
        self.client.post("/predict", json={"feature1": 1, "feature2": 2})
