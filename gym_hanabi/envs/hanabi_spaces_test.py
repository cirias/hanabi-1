import unittest

from gym_hanabi.envs import hanabi
from gym_hanabi.envs import hanabi_config
from gym_hanabi.envs import hanabi_spaces

class TestNestedSpaces(unittest.TestCase):
    def setUp(self):
        self.config = hanabi_config.MINI_HANABI_CONFIG
        self.spaces = hanabi_spaces.NestedSpaces(self.config)

        self.R1 = hanabi.Card("red", 1)
        self.R2 = hanabi.Card("red", 2)
        self.R3 = hanabi.Card("red", 3)
        self.G1 = hanabi.Card("green", 1)
        self.G2 = hanabi.Card("green", 2)
        self.G3 = hanabi.Card("green", 3)
        self.B1 = hanabi.Card("blue", 1)
        self.B2 = hanabi.Card("blue", 2)
        self.B3 = hanabi.Card("blue", 3)

        self.cards_samples = [
            (self.R1, (0, 0)), (self.R2, (0, 1)), (self.R3, (0, 2)),
            (self.G1, (1, 0)), (self.G2, (1, 1)), (self.G3, (1, 2)),
            (self.B1, (2, 0)), (self.B2, (2, 1)), (self.B3, (2, 2)),
            (None, (3, 3)),
        ]

        self.infos_samples = [
            (hanabi.Information("red", 1), (0, 0)),
            (hanabi.Information("red", 2), (0, 1)),
            (hanabi.Information("red", 3), (0, 2)),
            (hanabi.Information("red", None), (0, 3)),
            (hanabi.Information("green", 1), (1, 0)),
            (hanabi.Information("green", 2), (1, 1)),
            (hanabi.Information("green", 3), (1, 2)),
            (hanabi.Information("green", None), (1, 3)),
            (hanabi.Information("blue", 1), (2, 0)),
            (hanabi.Information("blue", 2), (2, 1)),
            (hanabi.Information("blue", 3), (2, 2)),
            (hanabi.Information("blue", None), (2, 3)),
            (hanabi.Information(None, 1), (3, 0)),
            (hanabi.Information(None, 2), (3, 1)),
            (hanabi.Information(None, 3), (3, 2)),
            (hanabi.Information(None, None), (3, 3)),
            (None, (4, 4)),
        ]

        self.actions_samples = [
            (hanabi.InformColorMove("red", 0), 0),
            (hanabi.InformColorMove("green", 0), 1),
            (hanabi.InformColorMove("blue", 0), 2),
            (hanabi.InformNumberMove(1, 0), 3),
            (hanabi.InformNumberMove(2, 0), 4),
            (hanabi.InformNumberMove(3, 0), 5),
            (hanabi.DiscardMove(0), 6),
            (hanabi.DiscardMove(1), 7),
            (hanabi.DiscardMove(2), 8),
            (hanabi.PlayMove(0), 9),
            (hanabi.PlayMove(1), 10),
            (hanabi.PlayMove(2), 11),
        ]

    def test_color_to_sample(self):
        self.assertEqual(self.spaces.color_to_sample("red"), 0)
        self.assertEqual(self.spaces.color_to_sample("green"), 1)
        self.assertEqual(self.spaces.color_to_sample("blue"), 2)

    def test_sample_to_color(self):
        self.assertEqual(self.spaces.sample_to_color(0), "red")
        self.assertEqual(self.spaces.sample_to_color(1), "green")
        self.assertEqual(self.spaces.sample_to_color(2), "blue")

    def test_number_to_sample(self):
        self.assertEqual(self.spaces.number_to_sample(1), 0)
        self.assertEqual(self.spaces.number_to_sample(2), 1)
        self.assertEqual(self.spaces.number_to_sample(3), 2)

    def test_sample_to_number(self):
        self.assertEqual(self.spaces.sample_to_number(0), 1)
        self.assertEqual(self.spaces.sample_to_number(1), 2)
        self.assertEqual(self.spaces.sample_to_number(2), 3)

    def test_card_to_sample(self):
        for card, sample in self.cards_samples:
            self.assertEqual(self.spaces.card_to_sample(card), sample)
            self.assertTrue(self.spaces.card_space().contains(sample))

    def test_sample_to_card(self):
        for card, sample in self.cards_samples:
            self.assertEqual(self.spaces.sample_to_card(sample), card)

    def test_cards_to_sample(self):
        discarded_cards = [self.R1, self.R1, self.R2, self.G1, self.B2]
        actual = self.spaces.cards_to_sample(discarded_cards)
        expected = (2, 1, 0, 1, 0, 0, 0, 1, 0)
        self.assertEqual(actual, expected)
        self.assertTrue(self.spaces.discarded_cards_space().contains(actual))

        played_cards = [self.R1, self.R2, self.G1, self.B2]
        actual = self.spaces.cards_to_sample(played_cards)
        expected = (1, 1, 0, 1, 0, 0, 0, 1, 0)
        self.assertEqual(actual, expected)
        self.assertTrue(self.spaces.played_cards_space().contains(actual))

    def test_sample_to_cards(self):
        actual = self.spaces.sample_to_cards((2, 1, 0, 1, 0, 0, 0, 1, 0))
        expected = [self.R1, self.R1, self.R2, self.G1, self.B2]
        self.assertEqual(actual, expected)

        actual = self.spaces.sample_to_cards((1, 1, 0, 1, 0, 0, 0, 1, 0))
        expected = [self.R1, self.R2, self.G1, self.B2]
        self.assertEqual(actual, expected)

    def test_information_to_sample(self):
        for info, sample in self.infos_samples:
            self.assertEqual(self.spaces.information_to_sample(info), sample)
            self.assertTrue(self.spaces.information_space().contains(sample))

    def test_sample_to_information(self):
        for info, sample in self.infos_samples:
            self.assertEqual(self.spaces.sample_to_information(sample), info)

    def test_action_to_sample(self):
        for action, sample in self.actions_samples:
            self.assertEqual(self.spaces.action_to_sample(action), sample)
            self.assertTrue(self.spaces.action_space().contains(sample))

    def test_sample_to_action(self):
        for action, sample in self.actions_samples:
            _cards = None
            self.assertEqual(self.spaces.sample_to_action(sample, _cards), action)

if __name__ == "__main__":
    unittest.main()
