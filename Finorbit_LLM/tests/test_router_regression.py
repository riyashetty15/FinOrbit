from backend.core.router import RouterAgent


def test_router_routes_plural_loans_to_credits_loans():
    router = RouterAgent()
    query = "Are NBFC loans always riskier than bank loans?"

    agent = router.route(query)
    intent = router.classify_query_intent(query)

    assert agent == "credits_loans"
    assert intent == "general"
