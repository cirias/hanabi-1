import collections
import gym
import gym.spaces
import gym.utils
import gym.utils.seeding
import termcolor

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
# Constants
################################################################################
MAX_TOKENS = 8
MAX_FUSES = 4
NUM_NUMBERS = 5
HAND_SIZE = 5
CARD_COUNTS = [3, 2, 2, 2, 1]

################################################################################
# Colors
################################################################################
class Colors(object):
    WHITE = "white"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    RED = "red"
    COLORS = [WHITE, YELLOW, GREEN, BLUE, RED]

COLOR_SPACE = gym.spaces.Discrete(len(Colors.COLORS))

def color_to_sample(color):
    """
    >>> color_to_sample(Colors.WHITE)
    0
    """
    assert (color in Colors.COLORS)
    return Colors.COLORS.index(color)

def sample_to_color(sample):
    """
    >>> sample_to_color(4)
    'red'
    """
    assert (0 <= sample <= len(Colors.COLORS))
    return Colors.COLORS[sample]

################################################################################
# Number
################################################################################
NUMBER_SPACE = gym.spaces.Discrete(NUM_NUMBERS)

def number_to_sample(number):
    """
    >>> number_to_sample(2)
    1
    """
    assert 1 <= number <= NUM_NUMBERS
    return number - 1

def sample_to_number(sample):
    """
    >>> sample_to_number(1)
    2
    """
    assert 0 <= sample <= (NUM_NUMBERS - 1)
    return sample + 1

################################################################################
# Cards
################################################################################
Card = collections.namedtuple('Card', ['color', 'number'])
CARD_SPACE = gym.spaces.MultiDiscrete([
    [0, len(Colors.COLORS) - 1], # Color
    [0, NUM_NUMBERS - 1]         # Number
])

def card_to_sample(card):
    """
    >>> card_to_sample(Card(Colors.WHITE, 1))
    [0, 0]
    >>> card_to_sample(Card(Colors.BLUE, 5))
    [3, 4]
    """
    return [color_to_sample(card.color), number_to_sample(card.number)]

def sample_to_card(sample):
    """
    >>> sample_to_card([0, 0])
    Card(color='white', number=1)
    >>> sample_to_card([3, 4])
    Card(color='blue', number=5)
    """
    assert (len(sample) == 2)
    return Card(sample_to_color(sample[0]), sample_to_number(sample[1]))

def render_card(card):
    s = "{}{}".format(card.number, card.color[0].upper())
    return termcolor.colored(s, card.color, attrs=["bold"])

################################################################################
# Discarded and played cards.
################################################################################
CARDS = [Card(c, n) for c in Colors.COLORS for n in range(1, NUM_NUMBERS + 1)]
DISCARDED_CARDS_SPACE = gym.spaces.MultiDiscrete([[0, 3] * len(CARDS)])
PLAYED_CARDS_SPACE = DISCARDED_CARDS_SPACE

def cards_to_sample(cards):
    """
    >>> cards = [Card(Colors.WHITE,4), Card(Colors.RED,2), Card(Colors.RED,2)]
    >>> cards_to_sample(cards)
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
    """
    assert (all(card in CARDS for card in CARDS))
    return [cards.count(card) for card in CARDS]

def sample_to_cards(sample):
    """
    >>> sample = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0]
    >>> sample_to_cards(sample) # doctest: +NORMALIZE_WHITESPACE
    [Card(color='white', number=4),
     Card(color='red', number=2),
     Card(color='red', number=2)]
    """
    assert (len(sample) == len(CARDS))
    return flatten([card] * count for (card, count) in zip(CARDS, sample))

################################################################################
# Information
################################################################################
Information = collections.namedtuple('Information', ['color', 'number'])
INFORMATION_SPACE = gym.spaces.MultiDiscrete([
    [0, len(Colors.COLORS)], # Color
    [0, NUM_NUMBERS]         # Number
])

def information_to_sample(info):
    """
    >>> information_to_sample(Information(Colors.WHITE, 4))
    [0, 3]
    >>> information_to_sample(Information(None, 4))
    [5, 3]
    >>> information_to_sample(Information(Colors.WHITE, None))
    [0, 5]
    >>> information_to_sample(Information(None, None))
    [5, 5]
    """
    color = info.color
    number = info.number
    color_sample = color_to_sample(color) if color else len(Colors.COLORS)
    number_sample = number_to_sample(number) if number else NUM_NUMBERS
    return [color_sample, number_sample]

def sample_to_information(sample):
    """
    >>> sample_to_information([0, 3])
    Information(color='white', number=4)
    >>> sample_to_information([5, 3])
    Information(color=None, number=4)
    >>> sample_to_information([0, 5])
    Information(color='white', number=None)
    >>> sample_to_information([5, 5])
    Information(color=None, number=None)
    """
    c = sample[0]
    n = sample[1]
    color = None if c == len(Colors.COLORS) else sample_to_color(c)
    number = None if n == NUM_NUMBERS else sample_to_number(n)
    return Information(color, number)

def render_information(info):
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

MOVES = ([InformColorMove(c) for c in Colors.COLORS] +
         [InformNumberMove(i) for i in range(1, NUM_NUMBERS + 1)] +
         [DiscardMove(i) for i in range(HAND_SIZE)] +
         [PlayMove(i) for i in range(HAND_SIZE)])
MOVE_SPACE = gym.spaces.Discrete(len(MOVES))

def move_to_sample(move):
    """
    >>> move_to_sample(InformColorMove(Colors.WHITE))
    0
    >>> move_to_sample(InformNumberMove(4))
    8
    """
    assert (move in MOVES)
    return MOVES.index(move)

def sample_to_move(sample):
    assert (0 <= sample <= len(MOVES))
    return MOVES[sample]

################################################################################
# Game state
################################################################################
class GameState(object):
    def __init__(self):
        self.num_tokens = MAX_TOKENS
        self.num_fuses = MAX_FUSES

        self.deck = [card for color in Colors.COLORS
                          for n, count in enumerate(CARD_COUNTS, 1)
                          for card in [Card(color, n)] * count]
        self.discarded_cards = []
        self.played_cards = dict() # color -> int

        self.ai_hand = tuple()
        self.ai_info = tuple()
        self.player_hand = tuple()
        self.player_info = tuple()

    def __repr__(self):
        return ("num_tokens:      {}\n".format(self.num_tokens) +
                "num_fuses:       {}\n".format(self.num_fuses) +
                "deck:            {}\n".format(self.deck) +
                "discarded_cards: {}\n".format(self.discarded_cards) +
                "played_cards:    {}\n".format(self.played_cards) +
                "ai_hand:         {}\n".format(self.ai_hand) +
                "ai_info:         {}\n".format(self.ai_info) +
                "player_hand:     {}\n".format(self.player_hand) +
                "player_info:     {}".format(self.player_info))

GAME_STATE_SPACE = gym.spaces.Tuple((
    gym.spaces.Discrete(MAX_TOKENS),                   # Tokens
    gym.spaces.Discrete(MAX_FUSES),                    # Fuses
    DISCARDED_CARDS_SPACE,                             # Discarded cards
    PLAYED_CARDS_SPACE,                                # Played cards
    gym.spaces.Tuple([CARD_SPACE] * HAND_SIZE),        # AI cards
    gym.spaces.Tuple([INFORMATION_SPACE] * HAND_SIZE), # AI info
    gym.spaces.Tuple([INFORMATION_SPACE] * HAND_SIZE)  # Player info
))

def game_state_to_sample(game_state):
    """
    >>> game_state = GameState()
    >>> game_state.ai_hand = [Card(Colors.WHITE, 1)] * 5
    >>> game_state.ai_info = [Information(None, None)] * 5
    >>> game_state.player_hand = [Card(Colors.WHITE, 1)] * 5
    >>> game_state.player_info = [Information(None, None)] * 5
    >>> game_state_to_sample(game_state) # doctest: +NORMALIZE_WHITESPACE
    (7,
     3,
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     ([0, 0], [0, 0], [0, 0], [0, 0], [0, 0]),
     ([5, 5], [5, 5], [5, 5], [5, 5], [5, 5]),
     ([5, 5], [5, 5], [5, 5], [5, 5], [5, 5]))
    """
    played_cards = [card for (color, x) in game_state.played_cards.items()
                         for card in [Card(color, y) for y in range(1, x + 1)]]
    return (
        game_state.num_tokens - 1,
        game_state.num_fuses - 1,
        cards_to_sample(game_state.discarded_cards),
        cards_to_sample(played_cards),
        tuple(card_to_sample(card) for card in game_state.ai_hand),
        tuple(information_to_sample(info) for info in game_state.ai_info),
        tuple(information_to_sample(info) for info in game_state.player_info),
    )

def sample_to_game_state(sample):
    """
    >>> sample = (7, 3,
    ...           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ...           [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ...           ([0,0], [0,0], [0,0], [0,0], [0,0]),
    ...           ([5,5], [5,5], [5,5], [5,5], [5,5]),
    ...           ([5,5], [5,5], [5,5], [5,5], [5,5]))
    >>> sample_to_game_state(sample) # doctest: +NORMALIZE_WHITESPACE
    num_tokens:      8
    num_fuses:       4
    deck: [Card(color='white', number=0), Card(color='white', number=0),
           Card(color='white', number=0), Card(color='white', number=1),
           Card(color='white', number=1), Card(color='white', number=2),
           Card(color='white', number=2), Card(color='white', number=3),
           Card(color='white', number=3), Card(color='white', number=4),
           Card(color='yellow', number=0), Card(color='yellow', number=0),
           Card(color='yellow', number=0), Card(color='yellow', number=1),
           Card(color='yellow', number=1), Card(color='yellow', number=2),
           Card(color='yellow', number=2), Card(color='yellow', number=3),
           Card(color='yellow', number=3), Card(color='yellow', number=4),
           Card(color='green', number=0), Card(color='green', number=0),
           Card(color='green', number=0), Card(color='green', number=1),
           Card(color='green', number=1), Card(color='green', number=2),
           Card(color='green', number=2), Card(color='green', number=3),
           Card(color='green', number=3), Card(color='green', number=4),
           Card(color='blue', number=0), Card(color='blue', number=0),
           Card(color='blue', number=0), Card(color='blue', number=1),
           Card(color='blue', number=1), Card(color='blue', number=2),
           Card(color='blue', number=2), Card(color='blue', number=3),
           Card(color='blue', number=3), Card(color='blue', number=4),
           Card(color='red', number=0), Card(color='red', number=0),
           Card(color='red', number=0), Card(color='red', number=1),
           Card(color='red', number=1), Card(color='red', number=2),
           Card(color='red', number=2), Card(color='red', number=3),
           Card(color='red', number=3), Card(color='red', number=4)]
    discarded_cards: []
    played_cards:    {}
    ai_hand:         (Card(color='white', number=1),
                      Card(color='white', number=1),
                      Card(color='white', number=1),
                      Card(color='white', number=1),
                      Card(color='white', number=1))
    ai_info:         (Information(color=None, number=None),
                      Information(color=None, number=None),
                      Information(color=None, number=None),
                      Information(color=None, number=None),
                      Information(color=None, number=None))
    player_hand:     ()
    player_info:     (Information(color=None, number=None),
                      Information(color=None, number=None),
                      Information(color=None, number=None),
                      Information(color=None, number=None),
                      Information(color=None, number=None))
    """
    game_state = GameState()
    game_state.num_tokens = sample[0] + 1
    game_state.num_fuses = sample[1] + 1
    game_state.discarded_cards = sample_to_cards(sample[2])
    game_state.played_cards = sample_to_cards(sample[3])
    game_state.ai_hand = tuple(sample_to_card(s) for s in sample[4])
    game_state.ai_info = tuple(sample_to_information(i) for i in sample[5])
    game_state.player_info = tuple(sample_to_information(i) for i in sample[6])
    return game_state

def render_game_state(gs):
    def render_cards(cards):
        return " ".join(render_card(card) for card in cards)

    def render_infos(infos):
        return " ".join(render_information(info) for info in infos)

    played_cards = []
    for c in Colors.COLORS:
        if c in gs.played_cards:
            played_cards.append(render_card(Card(c, gs.played_cards[c])))
        else:
            played_cards.append(termcolor.colored("--", c))

    return ("deck:      {}\n".format(len(gs.deck)) +
            "tokens:    {}/{}\n".format(gs.num_tokens, MAX_TOKENS) +
            "fuses:     {}/{}\n".format(gs.num_fuses, MAX_FUSES) +
            "discarded: {}\n".format(render_cards(gs.discarded_cards)) +
            "played:    {}\n".format(" ".join(played_cards)) +
            "-------------------------\n" +
            "AI hand:   {}\n".format(render_cards(gs.ai_hand)) +
            "AI info:   {}\n".format(render_infos(gs.ai_info)) +
            "hand:      {}\n".format(render_cards(gs.player_hand)) +
            "info:      {}".format(render_infos(gs.player_info)))

################################################################################
# Environment
################################################################################
class HanabiEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self._seed()
        self.action_space = MOVE_SPACE
        self.observation_space = GAME_STATE_SPACE
        self.reward_range = (0, 1)

    def _step(self, action):
        # TODO: Return (observation, reward, done, info).
        return (None, 0, False, None)

    def _reset(self):
        gs = GameState()
        self.np_random.shuffle(gs.deck)
        gs.ai_hand = [gs.deck.pop() for _ in range(HAND_SIZE)]
        gs.ai_info = [Information(None, None) for _ in range(HAND_SIZE)]
        gs.player_hand = [gs.deck.pop() for _ in range(HAND_SIZE)]
        gs.player_info = [Information(None, None) for _ in range(HAND_SIZE)]
        self.game_state = gs
        return game_state_to_sample(self.game_state)

    def _render(self, mode='human', close=False):
        if mode == "human":
            print(render_game_state(self.game_state))
        else:
            super(HanabiEnv, self).render(mode=mode)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
