# ============================================
# Local GenAI Churn Explanation using Ollama
# ============================================

import requests
import json


def generate_churn_explanation(
    customer_data,
    churn_probability
):

    prompt = f"""
    You are an AI telecom retention expert.

    Analyze the following customer profile
    and explain why the customer may churn.

    Customer Data:
    {customer_data}

    Churn Probability:
    {churn_probability:.2%}

    Provide:
    1. Main churn drivers
    2. Risk interpretation
    3. Retention suggestions

    Keep the response concise and business-friendly.
    """

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={
                "Content-Type": "application/json"
            },
            data=json.dumps(payload)
        )

        result = response.json()

        return result["response"]

    except Exception as e:

        return f"Error generating explanation: {str(e)}"