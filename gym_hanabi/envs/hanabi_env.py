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
    >>> [color_to_sample(c) for c in Colors.COLORS]
    [0, 1, 2, 3, 4]
    """
    assert (color in Colors.COLORS)
    return Colors.COLORS.index(color)

def sample_to_color(sample):
    """
    >>> [sample_to_color(s) for s in [0, 1, 2, 3, 4]]
    ['white', 'yellow', 'green', 'blue', 'red']
    """
    assert (0 <= sample <= len(Colors.COLORS))
    return Colors.COLORS[sample]

################################################################################
# Number
################################################################################
NUMBER_SPACE = gym.spaces.Discrete(NUM_NUMBERS)

def number_to_sample(number):
    """
    >>> [number_to_sample(x) for x in [1, 2, 3, 4, 5]]
    [0, 1, 2, 3, 4]
    """
    assert 1 <= number <= NUM_NUMBERS
    return number - 1

def sample_to_number(sample):
    """
    >>> [sample_to_number(s) for s in [0, 1, 2, 3, 4]]
    [1, 2, 3, 4, 5]
    """
    assert 0 <= sample < NUM_NUMBERS
    return sample + 1

################################################################################
# Cards
################################################################################
Card = collections.namedtuple('Card', ['color', 'number'])
CARD_SPACE = gym.spaces.Tuple((
    gym.spaces.Discrete(len(Colors.COLORS) + 1), # Color
    gym.spaces.Discrete(NUM_NUMBERS + 1)         # Number
))

def card_to_sample(card):
    """
    >>> cards = [Card(Colors.WHITE, x) for x in [1, 2, 3, 4, 5]]
    >>> [card_to_sample(card) for card in cards]
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
    >>> cards = [Card(Colors.RED, x) for x in [1, 2, 3, 4, 5]]
    >>> [card_to_sample(card) for card in cards]
    [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]
    >>> card_to_sample(None)
    [5, 5]
    """
    if card is None:
        return [len(Colors.COLORS), NUM_NUMBERS]
    return [color_to_sample(card.color), number_to_sample(card.number)]

def sample_to_card(sample):
    """
    >>> sample_to_card([0, 0])
    Card(color='white', number=1)
    >>> sample_to_card([0, 1])
    Card(color='white', number=2)
    >>> sample_to_card([4, 4])
    Card(color='red', number=5)
    >>> sample_to_card([5, 5])
    """
    assert len(sample) == 2
    if sample == card_to_sample(None):
        return None
    return Card(sample_to_color(sample[0]), sample_to_number(sample[1]))

def render_card(card):
    if card is None:
        return "  "
    s = "{}{}".format(card.number, card.color[0].upper())
    return termcolor.colored(s, card.color, attrs=["bold"])

################################################################################
# Discarded and played cards.
################################################################################
CARDS = [Card(c, n) for c in Colors.COLORS for n in range(1, NUM_NUMBERS + 1)]
DISCARDED_CARDS_SPACE = \
    gym.spaces.Tuple(tuple([gym.spaces.Discrete(4)] * len(CARDS)))
PLAYED_CARDS_SPACE = DISCARDED_CARDS_SPACE

def cards_to_sample(cards):
    """
    >>> cards = [Card(Colors.WHITE,4), Card(Colors.RED,2), Card(Colors.RED,2)]
    >>> cards_to_sample(cards)
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
    """
    assert all(card in CARDS for card in CARDS)
    return [cards.count(card) for card in CARDS]

def sample_to_cards(sample):
    """
    >>> sample = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0]
    >>> sample_to_cards(sample) # doctest: +NORMALIZE_WHITESPACE
    [Card(color='white', number=4),
     Card(color='red', number=2),
     Card(color='red', number=2)]
    """
    assert len(sample) == len(CARDS)
    return flatten([card] * count for (card, count) in zip(CARDS, sample))

################################################################################
# Information
################################################################################
Information = collections.namedtuple('Information', ['color', 'number'])
INFORMATION_SPACE = gym.spaces.Tuple((
    gym.spaces.Discrete(len(Colors.COLORS) + 2), # Color
    gym.spaces.Discrete(NUM_NUMBERS + 2)         # Number
))

def information_to_sample(info):
    """
    >>> information_to_sample(Information(Colors.WHITE, 1))
    [0, 0]
    >>> information_to_sample(Information(Colors.RED, 5))
    [4, 4]
    >>> information_to_sample(Information(None, 1))
    [5, 0]
    >>> information_to_sample(Information(Colors.WHITE, None))
    [0, 5]
    >>> information_to_sample(Information(None, None))
    [5, 5]
    >>> information_to_sample(None)
    [6, 6]
    """
    if info is None:
        return [len(Colors.COLORS) + 1, NUM_NUMBERS + 1]
    return [color_to_sample(info.color) if info.color else len(Colors.COLORS),
            number_to_sample(info.number) if info.number else NUM_NUMBERS]

def sample_to_information(s):
    """
    >>> sample_to_information([0, 0])
    Information(color='white', number=1)
    >>> sample_to_information([4, 4])
    Information(color='red', number=5)
    >>> sample_to_information([5, 0])
    Information(color=None, number=1)
    >>> sample_to_information([0, 5])
    Information(color='white', number=None)
    >>> sample_to_information([5, 5])
    Information(color=None, number=None)
    >>> sample_to_information([6, 6])
    """
    if s == information_to_sample(None):
        return None
    color = None if s[0] == len(Colors.COLORS) else sample_to_color(s[0])
    number = None if s[1] == NUM_NUMBERS else sample_to_number(s[1])
    return Information(color, number)

def render_information(info):
    if info is None:
        return "  "
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
    >>> move_to_sample(InformNumberMove(1))
    5
    >>> move_to_sample(DiscardMove(0))
    10
    >>> move_to_sample(PlayMove(0))
    15
    """
    offsets = [0, len(Colors.COLORS), NUM_NUMBERS, HAND_SIZE]
    if isinstance(move, InformColorMove):
        return sum(offsets[:1]) + Colors.COLORS.index(move.color)
    elif isinstance(move, InformNumberMove):
        return sum(offsets[:2]) + range(1, NUM_NUMBERS + 1).index(move.number)
    elif isinstance(move, DiscardMove):
        return sum(offsets[:3]) + range(HAND_SIZE).index(move.index)
    elif isinstance(move, PlayMove):
        return sum(offsets[:4]) + range(HAND_SIZE).index(move.index)
    else:
        raise ValueError("Unexpected move {}.".format(move))

def sample_to_move(sample):
    """
    >>> sample_to_move(0)
    InformColorMove(color='white')
    >>> sample_to_move(5)
    InformNumberMove(number=1)
    >>> sample_to_move(10)
    DiscardMove(index=0)
    >>> sample_to_move(15)
    PlayMove(index=0)
    """
    assert 0 <= sample <= len(MOVES)
    return MOVES[sample]

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
    def __init__(self):
        self.num_tokens = MAX_TOKENS
        self.num_fuses = MAX_FUSES
        self.deck = [card for color in Colors.COLORS
                          for n, count in enumerate(CARD_COUNTS, 1)
                          for card in [Card(color, n)] * count]
        self.discarded_cards = []
        self.played_cards = collections.defaultdict(int)
        self.ai = Hand([None] * HAND_SIZE, [None] * HAND_SIZE)
        self.player = Hand([None] * HAND_SIZE, [None] * HAND_SIZE)
        self.num_turns_left = -1
        self.player_turn = True

    def __repr__(self):
        return ("num_tokens:      {}\n".format(self.num_tokens) +
                "num_fuses:       {}\n".format(self.num_fuses) +
                "deck:            {}\n".format(self.deck) +
                "discarded_cards: {}\n".format(self.discarded_cards) +
                "played_cards:    {}\n".format(self.played_cards) +
                "ai_cards:        {}\n".format(self.ai.cards) +
                "ai.info:         {}\n".format(self.ai.info) +
                "player_cards     {}\n".format(self.player.cards) +
                "player.info:     {}".format(self.player.info))

    def current_reward(self):
        if self.num_fuses == 0:
            return 0
        return sum(self.played_cards.values())

    def remove_card(self, who, index):
        card = who.cards.pop(index)
        who.info.pop(index)
        assert card is not None
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
        if isinstance(move, InformColorMove) or isinstance(move, InformNumberMove):
            who = self.ai if self.player_turn else self.player
            self.play_information_move(who, move)
        elif isinstance(move, DiscardMove):
            who = self.player if self.player_turn else self.ai
            card = self.remove_card(who, move.index)
            self.discarded_cards.append(card)
            if self.num_tokens < MAX_TOKENS:
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

        # The game is over if there are no turns left.
        if self.num_fuses == 0:
            # We used up all the fuses.
            self.num_turns_left = 0
        elif all(self.played_cards[color] == len(CARD_COUNTS) for color in Colors.COLORS):
            # We got all fives!
            self.num_turns_left = 0
        elif len(self.deck) == 0:
            # There are no more cards in the deck. Start the last full round of
            # 2 players if we haven't already.
            if self.num_turns_left == -1:
                self.num_turns_left = 2
            else:
                self.num_turns_left -= 1

        self.player_turn = not self.player_turn

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
    >>> game_state.ai.cards = [Card(Colors.WHITE, 1)] * 5
    >>> game_state.ai.info = [Information(None, None)] * 5
    >>> game_state.player.cards = [Card(Colors.WHITE, 1)] * 5
    >>> game_state.player.info = [Information(None, None)] * 5
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
    other_player = game_state.ai if game_state.player_turn else game_state.player
    return (
        game_state.num_tokens - 1,
        game_state.num_fuses - 1,
        cards_to_sample(game_state.discarded_cards),
        cards_to_sample(played_cards),
        tuple(card_to_sample(card) for card in other_player.cards),
        tuple(information_to_sample(info) for info in game_state.ai.info),
        tuple(information_to_sample(info) for info in game_state.player.info),
    )

def render_game_state(gs):
    def render_cards(cards, show=True):
        shown_cards = " ".join(render_card(card) for card in cards)
        hidden_cards = " ".join("??" for card in cards)
        return shown_cards if show else hidden_cards

    def render_infos(infos):
        return " ".join(render_information(info) for info in infos)

    played_cards = []
    for c in Colors.COLORS:
        if gs.played_cards[c] > 0:
            played_cards.append(render_card(Card(c, gs.played_cards[c])))
        else:
            played_cards.append(termcolor.colored("--", c))

    them = gs.ai if gs.player_turn else gs.player
    you = gs.player if gs.player_turn else gs.ai

    return ("deck:       {}\n".format(len(gs.deck)) +
            "tokens:     {}/{}\n".format(gs.num_tokens, MAX_TOKENS) +
            "fuses:      {}/{}\n".format(gs.num_fuses, MAX_FUSES) +
            "discarded:  {}\n".format(render_cards(gs.discarded_cards)) +
            "played:     {}\n".format(" ".join(played_cards)) +
            "-------------------------\n" +
            "their hand: {}\n".format(render_cards(them.cards, show=True)) +
            "their info: {}\n".format(render_infos(them.info)) +
            "your hand:  {}\n".format(render_cards(you.cards, show=False)) +
            "your info:  {}".format(render_infos(you.info)))

################################################################################
# Environment
################################################################################
def random_policy(observation):
    return MOVE_SPACE.sample()

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

    def __init__(self, ai_policy=random_policy):
        self._seed()
        self.action_space = MOVE_SPACE
        self.observation_space = GAME_STATE_SPACE
        self.reward_range = (0, 1)
        self.ai_policy = ai_policy
        self._reset()

    def _step(self, action):
        raise NotImplementedError()

    def _reset(self):
        gs = GameState()
        self.np_random.shuffle(gs.deck)
        gs.ai.cards = [gs.deck.pop() for _ in range(HAND_SIZE)]
        gs.ai.info = [Information(None, None) for _ in range(HAND_SIZE)]
        gs.player.cards = [gs.deck.pop() for _ in range(HAND_SIZE)]
        gs.player.info = [Information(None, None) for _ in range(HAND_SIZE)]
        self.game_state = gs
        return game_state_to_sample(self.game_state)

    def _render(self, mode='human', close=False):
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
