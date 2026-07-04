import unittest
import numpy as np
from src.simulation import TournamentSimulator

class TestTournamentSimulation(unittest.TestCase):
    def setUp(self):
        self.simulator = TournamentSimulator("config.yaml")

    def test_match_probability_sum(self):
        # Probabilities should sum to exactly 1.0
        pa, pd, pb = self.simulator.predict_match_probabilities("Argentina", "France")
        self.assertAlmostEqual(pa + pd + pb, 1.0, places=5)

    def test_knockout_resolves_draw(self):
        # Test that is_knockout=True never returns "Draw"
        for _ in range(50):
            winner, goals_a, goals_b = self.simulator.simulate_match_outcome("Argentina", "France", is_knockout=True)
            self.assertNotEqual(winner, "Draw")
            self.assertIn(winner, ["Argentina", "France"])

    def test_group_stage_table(self):
        # Test group stage table sorting and shape
        group_tables = self.simulator.simulate_group_stage()
        
        # Verify all 12 groups are present
        self.assertEqual(len(group_tables), 12)
        
        # Verify Group A has 4 teams
        table_a = group_tables["A"]
        self.assertEqual(len(table_a), 4)
        
        # Check sorting: first team must have >= points than second team
        self.assertTrue(table_a[0][1] >= table_a[1][1])
        self.assertTrue(table_a[1][1] >= table_a[2][1])

    def test_full_tournament_simulation(self):
        # Test running a full tournament
        champ = self.simulator.simulate_tournament()
        
        # Gather all flat teams in 2026 WC
        all_teams = []
        for teams in self.simulator.groups_2026.values():
            all_teams.extend(teams)
            
        # Champion must be one of the participating teams
        self.assertIn(champ, all_teams)

if __name__ == '__main__':
    unittest.main()
