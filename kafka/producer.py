from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

contracts = [
    "Month-to-month",
    "One year",
    "Two year"
]

while True:

    customer_event = {

        "gender": random.choice(["Male", "Female"]),

        "SeniorCitizen": random.choice([0, 1]),

        "tenure": random.randint(1, 72),

        "MonthlyCharges": round(
            random.uniform(20, 120),
            2
        ),

        "Contract": random.choice(contracts)
    }

    producer.send(
        'customer-churn',
        value=customer_event
    )

    print("\nEvent Sent:")
    print(customer_event)

    time.sleep(5)