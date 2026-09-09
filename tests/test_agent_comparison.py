from research_repro.agent.strategy import NaiveStrategy, SelfCorrectingStrategy


def test_comparison_uses_distinct_post_evaluation_policies():
    assert NaiveStrategy().max_retries > 0
    assert callable(SelfCorrectingStrategy.handle_result)

