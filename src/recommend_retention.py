# ============================================
# Retention Recommendation Engine
# ====
def generate_retention_recommendation(
    churn_probability,
    contract_type,
    tenure,
    monthly_charges,
    total_services
):
    """
    Generate business retention strategy
    """

    recommendations = []


    # High Risk Customers
    if churn_probability >= 0.75:

        recommendations.append(
            "Priority retention outreach"
        )

        recommendations.append(
            "Assign customer success manager"
        )


    # Contract-based recommendation
    if contract_type == 'Month-to-month':

        recommendations.append(
            "Offer annual contract discount"
        )


    # New customer churn risk
    if tenure <= 12:

        recommendations.append(
            "Launch onboarding engagement campaign"
        )


    # Price sensitivity
    if monthly_charges >= 80:

        recommendations.append(
            "Provide personalized pricing offer"
        )


    # Low service adoption
    if total_services <= 2:

        recommendations.append(
            "Recommend bundled services"
        )


    # Fallback
    if len(recommendations) == 0:

        recommendations.append(
            "Standard retention monitoring"
        )


    return recommendations