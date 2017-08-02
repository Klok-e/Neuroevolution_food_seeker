import pygame
import math
import random

BLACK = (0, 0, 0)
ORANGE = (255, 174, 53)
GREEN = (43, 255, 13)
DARKGREEN = (0, 134, 6)
GRAY = (218, 220, 214)
SCREENRECT = pygame.Rect(0, 0, 800, 600)
GROUND_SIZE = (500, 500)
SIZE_OF_SEEKER = 5
FPS = 30
START_POPULATION = 100


class Camera():
    CAMERA_STEP = 20

    def __init__(self, startpos=(0, 0)):
        self.pos = (startpos[0], startpos[1])
        self.movingx = 0
        self.movingy = 0

    def move(self):
        self.pos = (self.pos[0] - self.movingx, self.pos[1] - self.movingy)

    def get_plgrsurf_pos(self):
        return (self.pos[0], self.pos[1])

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.movingy = -self.CAMERA_STEP
            if event.key == pygame.K_DOWN:
                self.movingy = self.CAMERA_STEP
            if event.key == pygame.K_LEFT:
                self.movingx = -self.CAMERA_STEP
            if event.key == pygame.K_RIGHT:
                self.movingx = self.CAMERA_STEP
        if event.type == pygame.KEYUP:
            self.movingy, self.movingx = 0, 0


class Vector2(object):
    def __init__(self, xy: tuple, angle=None):
        self.xy = xy

        self.angle_rad = math.atan2(self.xy[1], self.xy[0])

        self.length = self.get_length()

        self.normalize()

    def __str__(self):
        return str(math.degrees(self.angle_rad)) + ' |||| ' + str(self.xy)

    def __add__(self, other):
        return self.__thisVectorPlusAnotherVector(other)

    def __mul__(self, other):
        x1, y1 = self.get_componenXY()
        x2, y2 = other.get_componenXY()
        scalar = x1 * x2 + y1 * y2
        return scalar

    def get_componenXY(self):
        return (self.xy[0], self.xy[1])

    def get_length(self):
        x, y = self.get_componenXY()
        length = math.sqrt(x ** 2 + y ** 2)
        return length

    def get_angle(self):
        return math.degrees(self.angle_rad)

    def get_angle_rad(self):
        return self.angle_rad

    def change_components_to(self, xy):
        self.xy = xy

        self.angle_rad = math.atan2(xy[1], xy[0])
        self.length = self.get_length()

    def change_angle_by(self, angle_indeg):
        angle = math.radians(angle_indeg)
        self.angle_rad += angle

        self.change_components_to((math.cos(self.angle_rad), math.sin(self.angle_rad)))

    def normalize(self):
        x, y = self.xy
        length = self.length
        if length != 0:
            normx, normy = x / length, y / length
            self.change_components_to((normx, normy))

    def __thisVectorPlusAnotherVector(self, vect):
        # convert to radians
        selfAngle = math.radians(self.angle_rad)
        vectAngle = math.radians(vect.angle)
        # first vector's components
        x = math.cos(selfAngle) * self.length
        y = math.sin(selfAngle) * self.length
        # second vector's components
        x1 = math.cos(vectAngle) * vect.value
        y1 = math.sin(vectAngle) * vect.value
        # sum
        xx = x + x1
        yy = y + y1
        '''
        # calculate angle and value
        angle = math.atan2(yy, xx)
        value = math.hypot(xx, yy)
        # convert to degrees
        angle = math.degrees(angle)'''
        return Vector2((xx, yy))

class AI():
    def __init__(self):
        pass
class Seeker(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.original_img=pygame.image.load('images\seeker.jpg').convert()
        self.original_img = pygame.transform.scale(self.original_img,
                                                   (SIZE_OF_SEEKER, SIZE_OF_SEEKER))
        self.image = self.original_img.copy()
        self.rect: pygame.Rect = self.original_img.get_rect(center=pos)
        self.pos = pos
        self.vector = Vector2((0, 1))
        self.speed = 0

    def rotate_to_selfangle(self):
        pos = self.pos

        img = pygame.transform.rotate(self.original_img, -self.vector.get_angle() - 90)
        self.image = img
        # self.image = pygame.transform.scale(img, (TILESIZE[0] * 2, TILESIZE[1] * 2))

        self.rect = img.get_rect()
        self.rect.center = pos

    def update(self):
        self.rotate_to_selfangle()
        self.pos = self.move()

    def move(self):
        dx, dy = self.vector.get_componenXY()

        sp = self.speed
        dx *= sp
        dy *= sp

        pos = self.pos
        newposx, newposy = pos[0] + dx, pos[1] + dy

        return (newposx, newposy)


def main():
    # initialization
    pygame.init()
    display_surf = pygame.display.set_mode(SCREENRECT.size)

    ground = pygame.Surface(GROUND_SIZE)
    ground.fill(GRAY)

    # groups
    seekers = pygame.sprite.Group()

    # create population
    for i in range(START_POPULATION):
        x, y = random.randrange(GROUND_SIZE[0]), random.randrange(GROUND_SIZE[1])
        seekers.add(Seeker((x, y)))

    # clock
    timer = pygame.time.Clock()

    # camera
    camera = Camera()

    # font for displaying text
    textobj = pygame.font.Font(None, 50)

    def physics_step():
        # move the camera
        camera.move()

        # update

    def drawing_step():

        # draw a ground
        display_surf.fill(DARKGREEN)
        display_surf.blit(ground, camera.get_plgrsurf_pos())

        #draw seekers
        seekers.draw(ground)

        # display fps
        fps = timer.get_fps()
        display_surf.blit(textobj.render(str(round(fps, 2)), True, ORANGE), (200, 10))

        # update the screen
        pygame.display.update(SCREENRECT)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return

            camera.handle_input(event)

        # mouse_pos = pygame.mouse.get_pos()
        drawing_step()
        physics_step()

        timer.tick(FPS)


if __name__ == '__main__':
    main()
