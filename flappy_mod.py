from itertools import cycle
import random
import sys
import pygame
from pygame.locals import *

FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    # (
    #     'assets/sprites/redbird-upflap.png',
    #     'assets/sprites/redbird-midflap.png',
    #     'assets/sprites/redbird-downflap.png',
    # ),
    # # blue bird
    # (
    #     'assets/sprites/bluebird-upflap.png',
    #     'assets/sprites/bluebird-midflap.png',
    #     'assets/sprites/bluebird-downflap.png',
    # ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    #'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    #'assets/sprites/pipe-red.png',
)


try:
    xrange
except NameError:
    xrange = range


def launch():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)
    
    #while True:
    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.flip(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hitmask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )



def initialSetup():
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    basex = 0

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0,0))
    SCREEN.blit(IMAGES['player'][playerIndex],
                (playerx, playery + playerShmVals['val']))
    SCREEN.blit(IMAGES['base'], (basex, BASEY))

    pygame.display.update()
    FPSCLOCK.tick(FPS)

    return {
        'playery': playery + playerShmVals['val'],
        'basex': basex,
        'playerIndexGen': playerIndexGen,
    }

def mainGameSetup(movementInfo):
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    dt = FPSCLOCK.tick(FPS)/1000
    pipeVelX = -128 * dt

    # player velocity, max velocity, downward acceleration, acceleration on flap
    playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playerAccY    =   1   # players downward acceleration
    playerRot     =  45   # player's rotation
    playerVelRot  =   3   # angular speed
    playerRotThr  =  20   # rotation threshold
    playerFlapAcc =  -9   # players speed on flapping
    playerFlapped = False # True when player flaps

    return {
        'playerFlapAcc': playerFlapAcc,
        'playerx': playerx,
        'playery': playery,
        'upperPipes': upperPipes,
        'lowerPipes': lowerPipes,
        'playerIndexGen': playerIndexGen,
        'baseShift' : baseShift,
        'playerVelRot' : playerVelRot,
        'playerMaxVelY': playerMaxVelY,
        'playerAccY': playerAccY,
        'pipeVelX': pipeVelX,
        'playerRotThr': playerRotThr,
        'basex' : basex,
        'playerRot': playerRot,
        'playerVelY': playerVelY,
        'score': score,
        'playerIndex': playerIndex,
        'loopIter': loopIter,
        'playerMinVelY': playerMinVelY,
        'playerFlapped': playerFlapped,
    }

def action(gameInfo, action):
    if (action):
        if gameInfo['playery'] > -2 * IMAGES['player'][0].get_height():
            gameInfo['playerVelY'] = gameInfo['playerFlapAcc']
            gameInfo['playerFlapped'] = True
            SOUNDS['wing'].play()

    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            return False

    return True

def keyInput(gameInfo):
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
            if gameInfo['playery'] > -2 * IMAGES['player'][0].get_height():
                gameInfo['playerVelY'] = gameInfo['playerFlapAcc']
                gameInfo['playerFlapped'] = True
                SOUNDS['wing'].play()

def quit():
    pygame.quit()

def mainGame(gameInfo):
    #action(gameInfo)

    # check for crash here
    crashTest = checkCrash({'x': gameInfo['playerx'], 'y': gameInfo['playery'], 'index': gameInfo['playerIndex']},
                            gameInfo['upperPipes'], gameInfo['lowerPipes'])
    if crashTest[0]:
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        return image_data, gameInfo, True

    # check for score
    playerMidPos = gameInfo['playerx'] + IMAGES['player'][0].get_width() / 2
    for pipe in gameInfo['upperPipes']:
        pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
        if pipeMidPos <= playerMidPos < pipeMidPos + 4:
            gameInfo['score'] += 1
            SOUNDS['point'].play()

    # playerIndex basex change
    if (gameInfo['loopIter'] + 1) % 3 == 0:
        gameInfo['playerIndex'] = next(gameInfo['playerIndexGen'])
    gameInfo['loopIter'] = (gameInfo['loopIter'] + 1) % 30
    gameInfo['basex'] = -((-gameInfo['basex'] + 100) % gameInfo['baseShift'])

    # rotate the player
    if gameInfo['playerRot'] > -90:
        gameInfo['playerRot'] -= gameInfo['playerVelRot']

    # player's movement
    if gameInfo['playerVelY'] < gameInfo['playerMaxVelY'] and not gameInfo['playerFlapped']:
        gameInfo['playerVelY'] += gameInfo['playerAccY']
    if gameInfo['playerFlapped']:
        gameInfo['playerFlapped'] = False

        # more rotation to cover the threshold (calculated in visible rotation)
        gameInfo['playerRot'] = 45

    playerHeight = IMAGES['player'][gameInfo['playerIndex']].get_height()
    gameInfo['playery'] += min(gameInfo['playerVelY'], BASEY - gameInfo['playery'] - playerHeight)

    # move pipes to left
    for uPipe, lPipe in zip(gameInfo['upperPipes'], gameInfo['lowerPipes']):
        uPipe['x'] += gameInfo['pipeVelX']
        lPipe['x'] += gameInfo['pipeVelX']

    # add new pipe when first pipe is about to touch left of screen
    if 3 > len(gameInfo['upperPipes']) > 0 and 0 < gameInfo['upperPipes'][0]['x'] < 5:
        newPipe = getRandomPipe()
        gameInfo['upperPipes'].append(newPipe[0])
        gameInfo['lowerPipes'].append(newPipe[1])

    # remove first pipe if its out of the screen
    if len(gameInfo['upperPipes']) > 0 and gameInfo['upperPipes'][0]['x'] < -IMAGES['pipe'][0].get_width():
        gameInfo['upperPipes'].pop(0)
        gameInfo['lowerPipes'].pop(0)

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0,0))

    for uPipe, lPipe in zip(gameInfo['upperPipes'], gameInfo['lowerPipes']):
        SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
        SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

    SCREEN.blit(IMAGES['base'], (gameInfo['basex'], BASEY))
    # print score so player overlaps the score
    showScore(gameInfo['score'])

    # Player rotation has a threshold
    visibleRot = gameInfo['playerRotThr']
    if gameInfo['playerRot'] <= gameInfo['playerRotThr']:
        visibleRot = gameInfo['playerRot']
    
    #playerSurface = pygame.transform.rotate(IMAGES['player'][gameInfo['playerIndex']], visibleRot)
    playerSurface = pygame.transform.rotate(IMAGES['player'][gameInfo['playerIndex']], 0)
    SCREEN.blit(playerSurface, (gameInfo['playerx'], gameInfo['playery']))

    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    pygame.display.update()
    FPSCLOCK.tick(FPS)

    return image_data, gameInfo, crashTest[0]



def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
         playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collides with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                      player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
