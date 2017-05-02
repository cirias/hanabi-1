import collections
import termcolor

################################################################################
# Types
################################################################################
Card = collections.namedtuple('Card', ['color', 'number'])
Information = collections.namedtuple('Information', ['color', 'number'])

InformColorMove = collections.namedtuple('InformColorMove', ['color'])
InformNumberMove = collections.namedtuple('InformNumberMove', ['number'])
DiscardMove = collections.namedtuple('DiscardMove', ['index'])
PlayMove = collections.namedtuple('PlayMove', ['index'])

class Hand(object):
    def __init__(self, cards, info):
        self.cards = cards
        self.info = info

    def __eq__(self, other):
        return self.cards == other.cards and self.info == other.info

Observation = collections.namedtuple(
    "Observation",
    ["num_tokens", "num_fuses", "discarded_cards", "played_cards", "them",
     "you"])

################################################################################
# Helper Functions
################################################################################
def render_card(card):
    if card is None:
        return "  "
    else:
        s = "{}{}".format(card.number, card.color[0].upper())
        return termcolor.colored(s, card.color, attrs=["bold"])

def render_information(info):
    if info is None:
        return "  "
    else:
        number = info.number or "?"
        color = info.color[0].upper() if info.color else "?"
        return "{}{}".format(number, color)

def render_cards(cards, show=True):
    shown_cards = " ".join(render_card(card) for card in cards)
    hidden_cards = " ".join("??" for card in cards if card is not None)
    return shown_cards if show else hidden_cards

def render_infos(infos):
    return " ".join(render_information(info) for info in infos)

################################################################################
# Game Logic
################################################################################
class GameState(object):
    def __init__(self, config, random):
        self.config = config
        self.random = random

        self.num_tokens = config.max_tokens
        self.num_fuses = config.max_fuses
        self.discarded_cards = []
        self.played_cards = collections.defaultdict(int)
        self.num_turns_left = -1
        self.player_turn = True
        self.last_player_move = None
        self.last_ai_move = None

        # Shuffle the deck.
        self.deck = [card for color in config.colors
                          for number, count in enumerate(config.card_counts, 1)
                          for card in [Card(color, number)] * count]
        random.shuffle(self.deck)

        # Deal to the AI.
        ai = Hand([], [])
        ai.cards = [self.deck.pop() for _ in range(config.hand_size)]
        ai.info = [Information(None, None) for _ in range(config.hand_size)]
        self.ai = ai

        # Deal to the player.
        player = Hand([], [])
        player.cards = [self.deck.pop() for _ in range(config.hand_size)]
        player.info = [Information(None, None) for _ in range(config.hand_size)]
        self.player = player

    def get_current_cards(self):
        if self.player_turn:
            return self.player.cards
        else:
            return self.ai.cards

    def current_score(self):
        return sum(self.played_cards.values())

    def max_score(self):
        return len(self.config.colors) * len(self.config.card_counts)

    def remove_card(self, who, index):
        if index >= len(who.cards):
            raise ValueError("Cannot remove non-existent card.")
        card = who.cards.pop(index)
        who.info.pop(index)
        if card is None:
            raise ValueError("Cannot remove non-existent card.")
        return card

    def deal_card(self, who):
        if len(self.deck) == 0:
            who.cards.append(None)
            who.info.append(None)
        else:
            who.cards.append(self.deck.pop())
            who.info.append(Information(None, None))

    def play_information_move(self, who, move):
        if self.num_tokens == 0:
            raise ValueError("No more information tokens left.")

        for i, card in enumerate(who.cards):
            if card is None:
                continue
            if isinstance(move, InformColorMove):
                if card.color == move.color:
                    who.info[i] = who.info[i]._replace(color=card.color)
            else:
                assert isinstance(move, InformNumberMove)
                if card.number == move.number:
                    who.info[i] = who.info[i]._replace(number=card.number)

        self.num_tokens -= 1

    def play_move(self, move):
        assert self.num_turns_left != 0

        # Record the move.
        if self.player_turn:
            self.last_player_move = move
        else:
            self.last_ai_move = move

        # Play the move.
        if isinstance(move, InformColorMove) or isinstance(move, InformNumberMove):
            who = self.ai if self.player_turn else self.player
            self.play_information_move(who, move)
        elif isinstance(move, DiscardMove):
            who = self.player if self.player_turn else self.ai
            card = self.remove_card(who, move.index)
            self.discarded_cards.append(card)
            if self.num_tokens < self.config.max_tokens:
                self.num_tokens += 1
            self.deal_card(who)
        elif isinstance(move, PlayMove):
            who = self.player if self.player_turn else self.ai
            card = self.remove_card(who, move.index)
            if card.number == self.played_cards[card.color] + 1:
                self.played_cards[card.color] += 1
            else:
                self.num_fuses -= 1
                self.discarded_cards.append(card)
            self.deal_card(who)
        else:
            raise ValueError("Unexpected move {}.".format(move))

        # Figure out when to end the game.
        if self.num_fuses == 0:
            # We used up all the fuses.
            self.num_turns_left = 0
        elif self.current_score() == self.max_score():
            # We played every card.
            self.num_turns_left = 0
        elif len(self.deck) == 0:
            # If there are no more cards in the deck, then we start the last
            # rounds of the game (if we haven't already).
            if self.num_turns_left == -1:
                self.num_turns_left = self.config.num_turns_after_last_deal
            else:
                self.num_turns_left -= 1

        self.player_turn = not self.player_turn

    def to_observation(self):
        them = self.ai if self.player_turn else self.player
        you = self.player if self.player_turn else self.ai
        return Observation(self.num_tokens, self.num_fuses,
                           self.discarded_cards, self.played_cards, them, you)

    def render(self):
        played_cards = []
        for c in self.config.colors:
            if self.played_cards[c] > 0:
                played_cards.append(render_card(Card(c, self.played_cards[c])))
            else:
                played_cards.append(termcolor.colored("--", c))

        ai_turn = " " if self.player_turn else "*"
        player_turn = "*" if self.player_turn else " "
        return (
            " last_player_move: {}\n".format(self.last_player_move) +
            " last_ai_move:     {}\n".format(self.last_ai_move) +
            " deck:             {}\n".format(len(self.deck)) +
            " tokens:           {}/{}\n".format(self.num_tokens, self.config.max_tokens) +
            " fuses:            {}/{}\n".format(self.num_fuses, self.config.max_fuses) +
            " discarded:        {}\n".format(render_cards(self.discarded_cards)) +
            " played:           {}\n".format(" ".join(played_cards)) +
            "{}ai hand:          {}\n".format(ai_turn, render_cards(self.ai.cards)) +
            " ai info:          {}\n".format(render_infos(self.ai.info)) +
            "{}player hand:      {}\n".format(player_turn, render_cards(self.player.cards)) +
            " player info:      {}".format(render_infos(self.player.info))
        )

