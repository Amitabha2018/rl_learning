import pygame
import sys
def load():
    # path of player with different states
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap.jpg',
            'assets/sprites/redbird-midflap.jpg',
            'assets/sprites/redbird-downflap.jpg'
    )

    # path of background
    BACKGROUND_PATH = 'assets/sprites/background-black.jpg'

    # path of pipe
    PIPE_PATH = 'assets/sprites/pipe-green.jpg'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/1.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/2.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/3.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/4.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/5.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/6.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/7.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/8.jpg').convert_alpha(),
        pygame.image.load('assets/sprites/9.jpg').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.jpg').convert_alpha()

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

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
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

    return IMAGES, SOUNDS, HITMASKS

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask
