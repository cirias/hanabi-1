import collections
import gym
import gym.spaces

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
MAX_FUSES = 4
MAX_TOKENS = 8
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

################################################################################
# Discarded and played cards.
################################################################################
CARDS = [Card(c, n) for c in Colors.COLORS for n in range(1, NUM_NUMBERS + 1)]
DISCARDED_SPACE = gym.spaces.MultiDiscrete([[0, 3] * len(CARDS)])
PLAYED_SPACE = DISCARDED_SPACE

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
# Environment
################################################################################
class HanabiEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        # TODO: Initialize the following:
        #   - action_space:      The Space object corresponding to valid actions
        #   - observation_space: The Space object corresponding to valid
        #                        observations
        #   - reward_range:      A tuple corresponding to the min and max
        #                        possible rewards
        pass


    def _step(self, action):
        # TODO: Return (observation, reward, done, info).
        pass

    def _reset(self):
        # TODO: Reset game and return initial observation.
        pass

    def _render(self, mode='human', close=False):
        if mode == "human":
            # TODO: Draw the state of the game.
            pass
        else:
            super(HanabiEnv, self).render(mode=mode)

    def _seed(self, seed):
        # TODO: Figure out what this method should do.
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
