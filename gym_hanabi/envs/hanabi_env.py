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
    gym.spaces.Tuple([CARD_SPACE] * HAND_SIZE),        # Their cards
    gym.spaces.Tuple([INFORMATION_SPACE] * HAND_SIZE), # Their info
    gym.spaces.Tuple([INFORMATION_SPACE] * HAND_SIZE)  # Your info
))

class GameStateObservation(object):

    def __init__(self, sample):
        (num_tokens, num_fuses, discarded_cards, played_cards, their_cards,
                their_info, your_info) = sample
        self.num_tokens = num_tokens
        self.num_fuses = num_fuses

        self.discarded_cards = sample_to_cards(discarded_cards)
        played_cards = sample_to_cards(played_cards)
        self.played_cards = collections.defaultdict(int)
        for color, number in played_cards:
            if number > self.played_cards[color]:
                self.played_cards[color] = number

        # Filter out cards that don't exist, for hands that are smaller than
        # the maximum hand size.
        self.them = Hand([sample_to_card(sample) for sample in their_cards],
                         [sample_to_information(sample) for sample in their_info])
        self.them.cards = [card for card in self.them.cards if card is not None]
        self.them.info = [info for info in self.them.info if info is not None]
        self.you = Hand([None] * HAND_SIZE,
                        [sample_to_information(sample) for sample in your_info])
        self.you.info = [info for info in self.you.info if info is not None]

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

    them = game_state.ai if game_state.player_turn else game_state.player
    you = game_state.player if game_state.player_turn else game_state.ai

    return (
        game_state.num_tokens - 1,
        game_state.num_fuses - 1,
        cards_to_sample(game_state.discarded_cards),
        cards_to_sample(played_cards),
        tuple(card_to_sample(card) for card in them.cards),
        tuple(information_to_sample(info) for info in them.info),
        tuple(information_to_sample(info) for info in you.info),
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
def compute_play_or_discard(all_info, played_cards):
    """
    Return cards that we think are playable, and the candidate for a discard.
    A card is playable if we have number information for it that matches a card
    that could be played next. The candidate to discard is a card whose
    duplicate has already been played. If we can't find such a card, then it is
    a card for which we have the least amount of information.

    TODO: Determine situations when we should play a card with only color
    information.
    TODO: Discard cards in FIFO order, if there are multiple with the least
    amount of information.
    """
    play_cards = []
    discard_card = None
    for card, info in enumerate(all_info):
        color, number = info

        if number is None:
            # Don't try to play a card with no number information. If that
            # color pile is already complete, discard it.
            if color is not None and played_cards[color] == len(CARD_COUNTS):
                discard_card = card
            continue

        if color is None:
            # If all color piles are already past this number, discard the
            # card.
            all_colors_played = True
            for color in Colors.COLORS:
                if played_cards[color] < number:
                    all_colors_played = False
            if all_colors_played:
                discard_card = card

            # Check if the card's number is playable with respect to any
            # one of the colors.
            match = False
            for color in Colors.COLORS:
                if number == played_cards[color] + 1:
                    match = True
            if match:
                # If the check succeeds, add this card to the cards we think
                # are playable.
                play_cards.append(card)
        elif number == played_cards[color] + 1:
            # If we have color information, check if the card's number is
            # playable. If the check succeeds, add this card to the to top of
            # the cards we think are playable.
            play_cards = [card] + play_cards
        elif number <= played_cards[color]:
            # If this color pile is already past this number, discard the card.
            discard_card = card

    # If we weren't able to find a card that we could definitely discard, then
    # discard one with the least amount of information.
    if discard_card is None:
        discard_card = 0
        for card, info in enumerate(all_info):
            color, number = info
            if (color <= all_info[discard_card].color and number
                    <= all_info[discard_card].number):
                # This card has the same or less information than the current card
                # we want to discard.
                discard_card = card

    return play_cards, discard_card


def compute_information(observation):
    """
    Return information moves that we think will be helpful, based on the cards
    that we think the other player will play or discard using
    compute_play_or_discard. This is added to in order:
    1) Color information about the cards that the other will play that will
    result in a fuse.
    2) Number information about the card that the other will discard, if the
    card is eventually playable and has no duplicates left.
    3) Number information about the cards that the other has color information
    about, if the card is playable.
    4) Information about the cards that the other has no information about, if
    the card is playable. If the other player has other cards with the same
    number and a different color, and no color information for those cards,
    then give color information for the card we want them to play.  Else, give
    number information.
    """
    information = []
    their_cards_to_play, their_card_to_discard = compute_play_or_discard(
            observation.them.info, observation.played_cards)
    # (1)
    # NOTE: Color information is sufficient to prevent the other player from
    # playing this card, since compute_play_or_discard only returns cards to
    # play if we have number information.
    for card in their_cards_to_play:
        color, number = observation.them.cards[card]
        if number != observation.played_cards[color] + 1:
            information.append(InformColorMove(color))
    if information:
        return information

    # (2)
    card = observation.them.cards[their_card_to_discard]
    if card.number > observation.played_cards[card.color]:
        discard_count = observation.discarded_cards.count(card)
        if discard_count + 1 == CARD_COUNTS[card.number - 1]:
            information.append(InformNumberMove(card.number))
    if information:
        return information

    # (3)
    color_info_cards = [card_index for card_index, info in
        enumerate(observation.them.info) if info.color is not None and
        info.number is None]
    for card_index in color_info_cards:
        card = observation.them.cards[card_index]
        if card.number == observation.played_cards[card.color] + 1:
            information.append(InformNumberMove(card.number))
    if information:
        return information

    # (4)
    no_info_cards = [card_index for card_index, info in
            enumerate(observation.them.info) if info.color is None and
            info.number is None]
    for card_index in no_info_cards:
        card = observation.them.cards[card_index]
        if card.number == observation.played_cards[card.color] + 1:
            # A duplicate has the same number but a different color.
            duplicates = [i for i, dup in enumerate(observation.them.cards) if
                    dup.number == card.number and dup.color != card.color]
            # Only give color information about the duplicate if they don't
            # already have information about its color.
            duplicates = [i for i in duplicates if
                    observation.them.info[i].color is None]
            if duplicates:
                information.append(InformColorMove(card.color))
            else:
                information.append(InformNumberMove(card.number))
    return information


def fixed_policy(observation_sample):
    observation = GameStateObservation(observation_sample)

    play_cards, discard_card = compute_play_or_discard(observation.you.info,
            observation.played_cards)
    # Pretend to apply the moves that we would play, so we don't inform the
    # other player about a card that we might already have.
    for card in play_cards:
        number_info, color_info = observation.you.info[card]
        if color_info is not None:
            # If we have color information about the card we'll play, that
            # color pile will definitely increase.
            observation.played_cards[color_info] += 1
        else:
            # If we don't have color information about the card that we'll
            # play, a color pile whose next number matches may increase. Apply
            # the hypothetical move if exactly one color pile matches.
            possible_colors = [color for color, played_number in
                    observation.played_cards.items() if number_info ==
                    played_number + 1]
            if len(possible_colors) == 1:
                observation.played_cards[possible_colors[0]] += 1

    if observation.num_tokens > 0:
        information = compute_information(observation)
    else:
        information = []

    if information:
        return information[0]
    elif play_cards:
        return PlayMove(play_cards[0])
    else:
        return DiscardMove(discard_card)

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

    def __init__(self, ai_policy=fixed_policy):
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
