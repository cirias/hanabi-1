import collections
import gym
import gym.spaces
import gym.utils
import gym.utils.seeding
import termcolor
import six

################################################################################
# Helper functions
################################################################################
def flatten(xss):
    """
    >>> flatten([])
    []
    >>> flatten([[1], [2, 3], [4, 5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    return [x for xs in xss for x in xs]

################################################################################
# Configuration
################################################################################
Config = collections.namedtuple('Config', [
    "colors",                    # string list,
    "max_tokens",                # int,
    "max_fuses",                 # int,
    "hand_size",                 # int,
    "card_counts",               # int list,
    "num_turns_after_last_deal", # int,
])

HANABI_CONFIG = Config(
    ["red", "green", "blue", "yellow", "white"], # colors
    8,                                           # max_tokens
    4,                                           # max_fuses
    5,                                           # hand_size
    [3, 2, 2, 2, 1],                             # card_counts
    2                                            # num_turns_after_last_deal
)

MINI_HANABI_CONFIG = Config(
    ["red", "green", "blue"], # colors
    6,                        # max_tokens
    3,                        # max_fuses
    3,                        # hand_size
    [2, 2, 1],                # card_counts
    2                         # num_turns_after_last_deal
)

################################################################################
# Colors
################################################################################
def color_space(config):
    return gym.spaces.Discrete(len(config.colors))

def color_to_sample(config, color):
    """
    >>> config = HANABI_CONFIG
    >>> [color_to_sample(config, c) for c in config.colors]
    [0, 1, 2, 3, 4]
    >>> config = MINI_HANABI_CONFIG
    >>> [color_to_sample(config, c) for c in config.colors]
    [0, 1, 2]
    """
    assert (color in config.colors)
    return config.colors.index(color)

def sample_to_color(config, sample):
    """
    >>> [sample_to_color(HANABI_CONFIG, s) for s in [0, 1, 2, 3, 4]]
    ['red', 'green', 'blue', 'yellow', 'white']
    >>> [sample_to_color(MINI_HANABI_CONFIG, s) for s in [0, 1, 2]]
    ['red', 'green', 'blue']
    """
    assert 0 <= sample < len(config.colors)
    return config.colors[sample]

################################################################################
# Number
################################################################################
def number_space(config):
    return gym.spaces.Discrete(len(config.card_counts))

def number_to_sample(config, number):
    """
    >>> [number_to_sample(HANABI_CONFIG, x) for x in [1, 2, 3, 4, 5]]
    [0, 1, 2, 3, 4]
    >>> [number_to_sample(MINI_HANABI_CONFIG, x) for x in [1, 2, 3]]
    [0, 1, 2]
    """
    assert 1 <= number <= len(config.card_counts)
    return number - 1

def sample_to_number(config, sample):
    """
    >>> [sample_to_number(HANABI_CONFIG, s) for s in [0, 1, 2, 3, 4]]
    [1, 2, 3, 4, 5]
    >>> [sample_to_number(HANABI_CONFIG, s) for s in [0, 1, 2]]
    [1, 2, 3]
    """
    assert 0 <= sample < len(config.card_counts)
    return sample + 1

################################################################################
# Cards
################################################################################
Card = collections.namedtuple('Card', ['color', 'number'])

def card_space(config):
    """
    Consider a config with the colors [red, green, blue] and numbers [1, 2, 3].
    The card space encodes all possible combinations of color and number:

        (red,  1) -> (0, 0)
        (blue, 3) -> (2, 2)

    It also encodes the possibility of a missing card, which is represented in
    Python as None:

        None -> (3, 3)
    """
    return gym.spaces.Tuple((
        gym.spaces.Discrete(len(config.colors) + 1),     # Color
        gym.spaces.Discrete(len(config.card_counts) + 1) # Number
    ))

def card_to_sample(config, card):
    """
    >>> cards = [Card('red', x) for x in [1, 2, 3, 4, 5]]
    >>> [card_to_sample(HANABI_CONFIG, card) for card in cards]
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    >>> cards = [Card('white', x) for x in [1, 2, 3, 4, 5]]
    >>> [card_to_sample(HANABI_CONFIG, card) for card in cards]
    [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
    >>> card_to_sample(HANABI_CONFIG, None)
    (5, 5)
    """
    if card is None:
        return (len(config.colors), len(config.card_counts))
    else:
        color_sample = color_to_sample(config, card.color)
        number_sample = number_to_sample(config, card.number)
        return (color_sample, number_sample)

def sample_to_card(config, sample):
    """
    >>> sample_to_card(HANABI_CONFIG, (0, 0))
    Card(color='red', number=1)
    >>> sample_to_card(HANABI_CONFIG, (0, 1))
    Card(color='red', number=2)
    >>> sample_to_card(HANABI_CONFIG, (4, 4))
    Card(color='white', number=5)
    >>> sample_to_card(HANABI_CONFIG, (5, 5))
    """
    assert len(sample) == 2
    if sample == card_to_sample(config, None):
        return None
    else:
        color = sample_to_color(config, sample[0])
        number = sample_to_number(config, sample[1])
        return Card(color, number)

def render_card(card):
    if card is None:
        return "  "
    else:
        s = "{}{}".format(card.number, card.color[0].upper())
        return termcolor.colored(s, card.color, attrs=["bold"])

################################################################################
# Discarded and played cards.
################################################################################
def unique_cards(config):
    """
    Returns a list of all the unique cards specified by `config`. The cards are
    ordered first by color and then by number.

    >>> unique_cards(MINI_HANABI_CONFIG) # doctest: +NORMALIZE_WHITESPACE
    [Card(color='red', number=1),
     Card(color='red', number=2),
     Card(color='red', number=3),
     Card(color='green', number=1),
     Card(color='green', number=2),
     Card(color='green', number=3),
     Card(color='blue', number=1),
     Card(color='blue', number=2),
     Card(color='blue', number=3)]
    """
    num_numbers = len(config.card_counts)
    return [Card(c, n) for c in config.colors for n in range(1, num_numbers + 1)]

def discarded_cards_space(config):
    """
    The discarded cards space specifies the number of each unique card that has
    been discarded. For example, imagine a config with colors [red, blue] and
    numbers [1, 2]. `unique_cards` returns [1R, 2R, 1B, 2B]. Our discarded
    cards space has four numbers; the first is the number of discarded 1R's,
    the second is the number of discarded 2R's, and so on.
    """
    count = lambda number: config.card_counts[number - 1]
    spaces = [gym.spaces.Discrete(count(c.number) + 1) for c in unique_cards(config)]
    return gym.spaces.Tuple(tuple(spaces))

def played_cards_space(config):
    """
    The played cards space is similar to the discarded cards space, except that
    a card can only be played at most once.
    """
    num_cards = len(unique_cards(config))
    return gym.spaces.Tuple(tuple([gym.spaces.Discrete(2)] * num_cards))

def cards_to_sample(config, cards):
    """
    >>> cards = [Card("red", 1), Card("blue", 3), Card("blue", 3)]
    >>> cards_to_sample(MINI_HANABI_CONFIG, cards)
    (1, 0, 0, 0, 0, 0, 0, 0, 2)
    """
    assert all(card in unique_cards(config) for card in cards)
    return tuple([cards.count(card) for card in unique_cards(config)])

def sample_to_cards(config, sample):
    """
    >>> sample = [1, 0, 0, 0, 0, 0, 0, 0, 2]
    >>> sample_to_cards(MINI_HANABI_CONFIG, sample) # doctest: +NORMALIZE_WHITESPACE
    [Card(color='red', number=1),
     Card(color='blue', number=3),
     Card(color='blue', number=3)]
    """
    ucs = unique_cards(config)
    assert len(sample) == len(ucs)
    return flatten([card] * count for (card, count) in zip(ucs, sample))

################################################################################
# Information
################################################################################
Information = collections.namedtuple('Information', ['color', 'number'])

def information_space(config):
    """
    Consider a config with colors [red, blue] and numbers [1, 2]. Information
    can be
      - any combination of the colors and numbers (e.g. 1R, 2B);
      - any combination of an unknown color and a known number (e.g. 1?, 2?);
      - any combination of an unknown number and a known color (e.g. ?R, ?B);
      - completely unkown (e.g. ??, ??); or
      - absent.

    TODO: Fix the bug that Larry pointed out.
    """
    return gym.spaces.Tuple((
        gym.spaces.Discrete(len(config.colors) + 2),     # Color
        gym.spaces.Discrete(len(config.card_counts) + 2) # Number
    ))

def information_to_sample(config, info):
    """
    >>> config = HANABI_CONFIG
    >>> information_to_sample(config, Information("red", 1))
    (0, 0)
    >>> information_to_sample(config, Information("white", 5))
    (4, 4)
    >>> information_to_sample(config, Information(None, 1))
    (5, 0)
    >>> information_to_sample(config, Information("red", None))
    (0, 5)
    >>> information_to_sample(config, Information(None, None))
    (5, 5)
    >>> information_to_sample(config, None)
    (6, 6)
    """
    if info is None:
        return (len(config.colors) + 1, len(config.card_counts) + 1)
    else:
        color_sample = (color_to_sample(config, info.color) if info.color
                        else len(config.colors))
        number_sample = (number_to_sample(config, info.number) if info.number
                         else len(config.card_counts))
        return (color_sample, number_sample)

def sample_to_information(config, s):
    """
    >>> sample_to_information(HANABI_CONFIG, (0, 0))
    Information(color='red', number=1)
    >>> sample_to_information(HANABI_CONFIG, (4, 4))
    Information(color='white', number=5)
    >>> sample_to_information(HANABI_CONFIG, (5, 0))
    Information(color=None, number=1)
    >>> sample_to_information(HANABI_CONFIG, (0, 5))
    Information(color='red', number=None)
    >>> sample_to_information(HANABI_CONFIG, (5, 5))
    Information(color=None, number=None)
    >>> sample_to_information(HANABI_CONFIG, (6, 6))
    """
    if s == information_to_sample(config, None):
        return None
    else:
        color = (None if s[0] == len(config.colors)
                 else sample_to_color(config, s[0]))
        number = (None if s[1] == len(config.card_counts)
                  else sample_to_number(config, s[1]))
        return Information(color, number)

def render_information(info):
    if info is None:
        return "  "
    else:
        number = info.number or "?"
        color = info.color[0].upper() if info.color else "?"
        return "{}{}".format(number, color)

################################################################################
# Moves
################################################################################
InformColorMove = collections.namedtuple('InformColorMove', ['color'])
InformNumberMove = collections.namedtuple('InformNumberMove', ['number'])
DiscardMove = collections.namedtuple('DiscardMove', ['index'])
PlayMove = collections.namedtuple('PlayMove', ['index'])

def moves(config):
    """Returns a list of all the possible moves."""
    num_numbers = len(config.card_counts)
    return ([InformColorMove(c) for c in config.colors] +
            [InformNumberMove(i) for i in range(1, num_numbers + 1)] +
            [DiscardMove(i) for i in range(config.hand_size)] +
            [PlayMove(i) for i in range(config.hand_size)])

def move_space(config):
    return gym.spaces.Discrete(len(moves(config)))

def move_to_sample(config, move):
    """
    >>> move_to_sample(HANABI_CONFIG, InformColorMove("red"))
    0
    >>> move_to_sample(HANABI_CONFIG, InformNumberMove(1))
    5
    >>> move_to_sample(HANABI_CONFIG, DiscardMove(0))
    10
    >>> move_to_sample(HANABI_CONFIG, PlayMove(0))
    15
    """
    num_numbers = len(config.card_counts)
    offsets = [0, len(config.colors), num_numbers, config.hand_size]
    if isinstance(move, InformColorMove):
        return sum(offsets[:1]) + config.colors.index(move.color)
    elif isinstance(move, InformNumberMove):
        return sum(offsets[:2]) + range(1, num_numbers + 1).index(move.number)
    elif isinstance(move, DiscardMove):
        return sum(offsets[:3]) + range(config.hand_size).index(move.index)
    elif isinstance(move, PlayMove):
        return sum(offsets[:4]) + range(config.hand_size).index(move.index)
    else:
        raise ValueError("Unexpected move {}.".format(move))

def sample_to_move(config, sample):
    """
    >>> sample_to_move(HANABI_CONFIG, 0)
    InformColorMove(color='red')
    >>> sample_to_move(HANABI_CONFIG, 5)
    InformNumberMove(number=1)
    >>> sample_to_move(HANABI_CONFIG, 10)
    DiscardMove(index=0)
    >>> sample_to_move(HANABI_CONFIG, 15)
    PlayMove(index=0)
    """
    assert 0 <= sample < len(moves(config))
    return moves(config)[sample]

################################################################################
# Hand
################################################################################
class Hand(object):
    def __init__(self, cards, info):
        self.cards = cards
        self.info = info

################################################################################
# Game state
################################################################################
class GameState(object):
    def __init__(self, config):
        self.config = config
        self.num_tokens = config.max_tokens
        self.num_fuses = config.max_fuses
        self.deck = [card for color in config.colors
                          for n, count in enumerate(config.card_counts, 1)
                          for card in [Card(color, n)] * count]
        self.discarded_cards = []
        self.played_cards = collections.defaultdict(int)
        self.num_turns_left = -1
        self.player_turn = True
        self.ai = Hand([None] * config.hand_size, [None] * config.hand_size)
        self.last_ai_move = None
        self.player = Hand([None] * config.hand_size, [None] * config.hand_size)
        self.last_player_move = None

    def current_reward(self):
        return sum(self.played_cards.values())

    def remove_card(self, who, index):
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
                    info = who.info[i]
                    who.info[i] = Information(card.color, info.number)
            else:
                assert isinstance(move, InformNumberMove)
                if card.number == move.number:
                    info = who.info[i]
                    who.info[i] = Information(info.color, card.number)

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
        max_reward = len(self.config.colors) * len(self.config.card_counts)
        if self.num_fuses == 0:
            # We used up all the fuses.
            self.num_turns_left = 0
        elif self.current_reward() == max_reward:
            # We got all fives!
            self.num_turns_left = 0
        elif len(self.deck) == 0:
            # There are no more cards in the deck. Start the last full round of
            # 2 players if we haven't already.
            if self.num_turns_left == -1:
                self.num_turns_left = self.config.num_turns_after_last_deal
            else:
                self.num_turns_left -= 1

        self.player_turn = not self.player_turn

def game_state_space(config):
    c = config
    return gym.spaces.Tuple((
        gym.spaces.Discrete(config.max_tokens),                 # Tokens
        gym.spaces.Discrete(config.max_fuses),                  # Fuses
        discarded_cards_space(c),                               # Discarded cards
        played_cards_space(c),                                  # Played cards
        gym.spaces.Tuple([card_space(c)] * c.hand_size),        # Their cards
        gym.spaces.Tuple([information_space(c)] * c.hand_size), # Their info
        gym.spaces.Tuple([information_space(c)] * c.hand_size)  # Your info
    ))

class GameStateObservation(object):
    def __init__(self, config, sample):
        (num_tokens, num_fuses, discarded_cards, played_cards, their_cards,
                their_info, your_info) = sample
        self.num_tokens = num_tokens
        self.num_fuses = num_fuses

        self.discarded_cards = sample_to_cards(config, discarded_cards)
        played_cards = sample_to_cards(config, played_cards)
        self.played_cards = collections.defaultdict(int)
        for color, number in played_cards:
            if number > self.played_cards[color]:
                self.played_cards[color] = number

        # Filter out cards that don't exist, for hands that are smaller than
        # the maximum hand size.
        self.them = Hand(
            [sample_to_card(config, sample) for sample in their_cards],
            [sample_to_information(config, sample) for sample in their_info])
        self.them.cards = [card for card in self.them.cards if card is not None]
        self.them.info = [info for info in self.them.info if info is not None]
        assert len(self.them.cards) == len(self.them.info)

        self.you = Hand(
            [None] * config.hand_size,
            [sample_to_information(config, sample) for sample in your_info])
        self.you.info = [info for info in self.you.info if info is not None]
        self.you.cards = [None] * len(self.you.info)

def game_state_to_sample(config, game_state):
    """
    >>> game_state = GameState(HANABI_CONFIG)
    >>> game_state.ai.cards = [Card("red", 1)] * 5
    >>> game_state.ai.info = [Information(None, None)] * 5
    >>> game_state.player.cards = [Card("white", 5)] * 5
    >>> game_state.player.info = [Information("white", None)] * 5
    >>> game_state_to_sample(HANABI_CONFIG, game_state) # doctest: +NORMALIZE_WHITESPACE
    (7,
     3,
     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
     ((0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
     ((5, 5), (5, 5), (5, 5), (5, 5), (5, 5)),
     ((4, 5), (4, 5), (4, 5), (4, 5), (4, 5)))
    """
    played_cards = [card for (color, count) in game_state.played_cards.items()
                         for card in [Card(color, n) for n in range(1, count + 1)]]

    them = game_state.ai if game_state.player_turn else game_state.player
    you = game_state.player if game_state.player_turn else game_state.ai

    return (
        game_state.num_tokens - 1,
        game_state.num_fuses - 1,
        cards_to_sample(config, game_state.discarded_cards),
        cards_to_sample(config, played_cards),
        tuple(card_to_sample(config, card) for card in them.cards),
        tuple(information_to_sample(config, info) for info in them.info),
        tuple(information_to_sample(config, info) for info in you.info),
    )

def render_game_state(gs):
    def render_cards(cards, show=True):
        shown_cards = " ".join(render_card(card) for card in cards)
        hidden_cards = " ".join("??" for card in cards if card is not None)
        return shown_cards if show else hidden_cards

    def render_infos(infos):
        return " ".join(render_information(info) for info in infos)

    played_cards = []
    for c in gs.config.colors:
        if gs.played_cards[c] > 0:
            played_cards.append(render_card(Card(c, gs.played_cards[c])))
        else:
            played_cards.append(termcolor.colored("--", c))

    ai_turn = " " if gs.player_turn else "*"
    player_turn = "*" if gs.player_turn else " "
    return (" last_player_move: {}\n".format(gs.last_player_move) +
            " last_ai_move:     {}\n".format(gs.last_ai_move) +
            " deck:             {}\n".format(len(gs.deck)) +
            " tokens:           {}/{}\n".format(gs.num_tokens, gs.config.max_tokens) +
            " fuses:            {}/{}\n".format(gs.num_fuses, gs.config.max_fuses) +
            " discarded:        {}\n".format(render_cards(gs.discarded_cards)) +
            " played:           {}\n".format(" ".join(played_cards)) +
            "{}ai hand:          {}\n".format(ai_turn, render_cards(gs.ai.cards)) +
            " ai info:          {}\n".format(render_infos(gs.ai.info)) +
            "{}player hand:      {}\n".format(player_turn, render_cards(gs.player.cards, show=False)) +
            " player info:      {}".format(render_infos(gs.player.info)))

################################################################################
# Environment
################################################################################
class HanabiEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def play_move(self, move):
        gs = self.game_state
        reward = -1 * gs.current_reward()
        gs.play_move(move)
        # The reward from this move is the total current reward, minus the
        # total reward from the previous game state.
        reward += gs.current_reward()

        # The game is over if there are no turns left.
        done = gs.num_turns_left == 0
        return reward, done

    def __init__(self, config, ai_policy=None):
        self.config = config
        self._seed()
        self.action_space = move_space(config)
        self.observation_space = game_state_space(config)
        self.ai_policy = ai_policy

    def _step(self, action):
        raise NotImplementedError()

    def _reset(self):
        gs = GameState(self.config)
        self.np_random.shuffle(gs.deck)
        hand_size = self.config.hand_size
        gs.ai.cards = [gs.deck.pop() for _ in range(hand_size)]
        gs.ai.info = [Information(None, None) for _ in range(hand_size)]
        gs.player.cards = [gs.deck.pop() for _ in range(hand_size)]
        gs.player.info = [Information(None, None) for _ in range(hand_size)]
        self.game_state = gs
        return game_state_to_sample(self.config, self.game_state)

    def _render(self, mode='human', close=False):
        if close:
            return

        if mode == "human":
            print(render_game_state(self.game_state))
        elif mode == "ansi":
            s = six.StringIO()
            s.write(render_game_state(self.game_state) + "\n")
            return s
        else:
            super(HanabiEnv, self).render(mode=mode)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
