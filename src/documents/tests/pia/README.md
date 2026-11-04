# PIA-tests (groep 2, Privacy and Big Data)

Tests die privacy-issues uit onze PIA aantonen. Use case: huisartsenpraktijk.
Alle data is fictief.

## Draaien

Vanuit de root van de repo:

    uv run pytest src/documents/tests/pia -n 0 --no-cov -v

Alle PIA-tests, ook als ze elders staan:

    uv run pytest -m pia -n 0 --no-cov -v

## Conventies

- Eén bestand per issue: `test_issueN_<onderwerp>.py`.
- Een test assert het **gewenste** privacygedrag.
- Toont de test een issue aan, markeer hem met `@pytest.mark.xfail(reason="PIA issue N: ...")`.
  `strict_xfail` staat aan in pyproject.toml: als het issue ooit opgelost is,
  wordt de test XPASS en faalt de run, zodat we het merken.
- Voeg waar mogelijk een controletest toe die slaagt, om te tonen dat de
  testopzet zelf klopt.
- Gebruik de fixtures uit `conftest.py` (`gp_a`, `gp_b`, `assistant`,
  `practice_manager`, `client_for`, de documenten) in plaats van eigen gebruikers.
