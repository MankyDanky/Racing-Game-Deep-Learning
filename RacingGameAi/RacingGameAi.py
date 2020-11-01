#Imports
import pygame
import neat
import pickle
import os
import math
pygame.font.init()

#Generation Variable
GEN = 0

#Define Window Size
WIN_WIDTH = 694
WIN_HEIGHT = 769

#Import Images
CAR_IMG = pygame.image.load(os.path.join("imgs", "Car.png"))
RAILINGS = pygame.image.load(os.path.join("imgs", "RailingSet.png"))
PROGRAM_ICON = pygame.image.load(os.path.join("imgs", "ProgramIcon.png"))
BACKGROUND_IMG = pygame.image.load(os.path.join("imgs", "Background.png"))
RAY_IMG = pygame.image.load(os.path.join("imgs", "Ray.png"))

#Import Font
STAT_FONT = pygame.font.Font("Hardpixel.OTF", 30)

#Configure Pygame Window
pygame.display.init()
pygame.display.set_caption("Racing Game Deep Learning")
pygame.display.set_icon(PROGRAM_ICON)

class Ray:
    #Define Constants
    IMG = RAY_IMG

    #Init Func
    def __init__(self, x, y, rot):
        self.x = x
        self.y = y
        self.rot = rot
        self.img = self.IMG

    # Draw Func
    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.img, self.rot)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    # Mask/Hitbox Func
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Car:
    #Define Constants
    IMG = CAR_IMG
    VEL = 3
    ROT_VEL = 3

    #Init Func
    def __init__(self, x, y, rot):
        self.x = x
        self.y = y
        self.rot = rot
        self.img = self.IMG
        self.vel = self.VEL
        self.rot_Vel = self.ROT_VEL
        #Create Rays
        self.ray = Ray(x, y, rot)
        self.ray1 = Ray(x, y, rot + 90)
        self.ray2 = Ray(x, y, rot + 270)
        self.ray3 = Ray(x, y, rot + 315)
        self.ray4 = Ray(x, y, rot + 45)

    #Move Func
    def move(self):
        #Calculate Move Direction
        moveY = math.cos(self.rot*math.pi/180)
        moveX = math.sin(self.rot*math.pi/180)

        moveXR = math.sin((self.rot+90)*math.pi/180)
        moveYR = math.cos((self.rot+90)*math.pi/180)

        #Move Car
        self.x += moveX*self.vel
        self.y += moveY*self.vel

        #Move Rays
        self.ray.x = self.x+moveX*48
        self.ray.y = self.y+moveY*48
        self.ray.rot = self.rot
        self.ray1.x = self.x+moveXR*40
        self.ray1.y = self.y+moveYR*40
        self.ray1.rot = self.rot+90
        self.ray2.x = self.x-moveXR*40
        self.ray2.y = self.y-moveYR*40
        self.ray2.rot = self.rot+270
        self.ray3.x = self.x-moveXR*32 + moveX*40
        self.ray3.y = self.y-moveYR*32 + moveY*40
        self.ray3.rot = self.rot+315
        self.ray4.x = self.x + moveXR * 32 + moveX*40
        self.ray4.y = self.y + moveYR * 32 + moveY*40
        self.ray4.rot = self.rot + 45

    #Rotate Func
    def rotate(self, dir):
        #Move Car According To Direction Parameter
        if dir == -1:
            self.rot -= self.rot_Vel
        if dir == 1:
            self.rot += self.rot_Vel

    #Draw Func
    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.img, self.rot)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    #Mask/Hitbox Func
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Railing:
    #Define Constants
    IMG = RAILINGS

    #Init Func
    def __init__(self):
        self.x = 0
        self.y = 0
        self.img = self.IMG

    #Draw Func
    def draw(self, win):
        win.blit(self.img, (0, 0))

    #Collide Func
    def collide(self, car):
        #Get Masks
        car_mask = car.get_mask()
        railing_mask = pygame.mask.from_surface(self.IMG)

        #Get Distance Between Bird And Pipes
        offset = (int(self.x - car.x), int(self.y - round(car.y)))

        #Check For Pixel Overlap
        point = car_mask.overlap(railing_mask, offset)

        #Return True Or False According To Above
        if point:
            return True, point
        return False, point

#Def Draw Window Func
def draw_window(win, cars, gen):
    #Set Background Color
    win.fill([48, 48, 48])

    #Draw Background
    win.blit(BACKGROUND_IMG, (0, 0))

    #Draw Cars
    for car in cars:
        car.draw(win)

    #Draw Gen
    genText = STAT_FONT.render("Gen: " + str(gen), 1, (255, 55, 55))
    win.blit(genText, (10, 2))

    #Update Display
    pygame.display.update()

#Def Main Func
def main(genomes, config):
    #Increase Generation Every Time We Run Main
    global GEN
    GEN += 1

    #Initiliaze Objects
    clock = pygame.time.Clock()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

    #Initiliaze Object Arrays
    nets = []
    ge = []
    cars = []
    railing = Railing()

    #Create Cars
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(38, 156, 0))
        g.fitness = 0
        ge.append(g)

    #Initiliaze Race Track
    #railingPositions = [(18, 81, 0), (18, 128, 0), (18, 175, 0), (18, 222, 0), (18, 269, 0), (18, 316, 0), (18, 363, 0), #Railings Going Down On Left Side
                        #(18, 410, 0), (18, 457, 0), (18, 504, 0), (18, 551, 0), (18, 598, 0), (18, 645, 0), #Railings Going Down On Left Side
                        #(79, 152, 0), (79, 199, 0), (79, 246, 0), (79, 293, 0), (79, 340, 0), #Railings Going Down On Right Side
                        #(79, 387, 0), (79, 434, 0), (79, 481, 0), (79, 528, 0), (79, 575, 0), #Railings Going Down On Right Side
                        #(35, 686, 45), (77, 703, 90), (124, 703, 90), (171, 703, 90), (218, 703, 90), (260, 686, 315), #Rotate To Railings Going Right On Bottom Side
                        #(95, 613, 45), (108, 626, 45), (148, 642, 90), (200, 613, 315), (187, 626, 315), #Rotate To Railings Going Right On Upper Bottom Side
                        #(277, 645, 0), (277, 598, 0), (277, 551, 0), (277, 504, 0), (277, 457, 0), (277, 410, 0), #Railings Going Up On Right Side
                        #(277, 363, 0), (277, 316, 0), (277, 269, 0), #Railings Going Up On Right Side
                        #(216, 575, 0), (216, 528, 0), (216, 481, 0), (216, 434, 0), (216, 387, 0), (216, 340, 0), #Railings Going Up On Left Side
                        #(216, 293, 0), (216, 246, 0), (216, 199, 0), #Railings Going Up On Left Side
                        #(233, 158, 315), (275, 141, 90), (322, 141, 90), (369, 141, 90), (416, 141, 90), (458, 158, 45), #Rotate To Railings Going Right On Top Side
                        #(293, 231, 315), (306, 218, 315), (345, 202, 90), (384, 218, 45), (397, 231, 45), #Rotate To Railings Going Right On Lower Top Side
                        #(474, 199, 0), (474, 246, 0), (474, 293, 0), (474, 340, 0), (474, 387, 0), (474, 434, 0), #Railings Going Down On Right Side
                        #(474, 481, 0), (474, 528, 0), (474, 575, 0), #Railings Going Down On Right Side
                        #(413, 269, 0), (413, 316, 0), (413, 363, 0), (413, 410, 0), (413, 457, 0), (413, 504, 0), #Railings Going Down On Left Side
                        #(413, 551, 0), (413, 598, 0), (413, 645, 0), #Railings Going Down On Left Side
                        #(430, 686, 45), (472, 703, 90), (519, 703, 90), (566, 703, 90), (613, 703, 90), (655, 686, 315), #Rotate To Railings Going Right On Bottom Side
                        #(490, 613, 45), (503, 626, 45), (543, 642, 90), (595, 613, 315), (582, 626, 315), #Rotate To Railings Going Right On Upper Bottom Side
                        #(672, 645, 0), (672, 598, 0), (672, 551, 0), (672, 504, 0), (672, 457, 0), (672, 410, 0), #Railings Going Up On Right Side
                        #(672, 363, 0), (672, 316, 0), (672, 269, 0), (672, 222, 0), (672, 175, 0), (672, 128, 0), (672, 81, 0), #Railings Going Up On Right Side
                        #(611, 575, 0), (611, 528, 0), (611, 481, 0), (611, 434, 0), (611, 387, 0), (611, 340, 0), #Railings Going Up On Left Side
                        #(611, 293, 0), (611, 246, 0), (611, 199, 0), (611, 152, 0), #Railings Going Up On Left Side
                        #(655, 40, 45), (77, 23, 90), (124, 23, 90), (171, 23, 90), (218, 23, 90), (265, 23, 90), #Rotate To Railings Going Left On Top Side
                        #(347, 23, 90), (312, 23, 90), (378, 23, 90), (425, 23, 90), (472, 23, 90), (519, 23, 90), #Rotate To Railings Going Left On Top Side
                        #(566, 23, 90), (613, 23, 90), (35, 40, 315), #Rotate To Railings Going Left On Top Side
                        #(543, 84, 90), (95, 113, 315), (108, 100, 315), (147, 84, 90), (194, 84, 90), (241, 84, 90), #Rotate To Railings Going Left On Lower Top Side
                        #(288, 84, 90), (335, 84, 90), (382, 84, 90), (429, 84, 90), (476, 84, 90), (523, 84, 90), #Rotate To Railings Going Left On Lower Top Side
                        #(595, 113, 45), (582, 100, 45)] #Rotate To Railings Going Left On Lower Top Side

    #Instantiate Cars And Railings
    #for r in railingPositions:
        #railings.append(Railing(r[0], r[1], r[2]))
    #Game Loop
    run = True
    while run:
        #Set Tickrate
        clock.tick(60)

        #Quit Game If X Is Clicked
        for event in pygame.event.get():
            #Check If X Was Clicked
            if event.type == pygame.QUIT:
                # Quit If X Was Clicked
                run = False
                pygame.quit()
                quit()

        #Check If All Cars Are Crashed
        if len(cars) == 0:
            run = False
            break

        # Move Cars
        for x, car in enumerate(cars):
            car.move()
            #Give Fitness Points For Staying Alive
            ge[x].fitness += 0.03

            #Calculate Inputs
            if not railing.collide(car.ray)[1] is None:
                xPos = car.ray.x + railing.collide(car.ray)[1][0]
                xDistance = math.fabs(car.x - xPos) - 10
            else:
                xDistance = 100

            if not railing.collide(car.ray)[1] is None:
                yPos = car.ray.y + railing.collide(car.ray)[1][1]
                yDistance = math.fabs(car.y - yPos) - 10
            else:
                yDistance = 100

            if not railing.collide(car.ray1)[1] is None:
                xPos1 = car.ray1.x + railing.collide(car.ray1)[1][0]
                xDistance1 = math.fabs(car.x - xPos1) - 10
            else:
                xDistance1 = 100

            if not railing.collide(car.ray1)[1] is None:
                yPos1 = car.ray1.y + railing.collide(car.ray1)[1][1]
                yDistance1 = math.fabs(car.y - yPos1) - 10
            else:
                yDistance1 = 100

            if not railing.collide(car.ray2)[1] is None:
                xPos2 = car.ray2.x + railing.collide(car.ray2)[1][0]
                xDistance2 = math.fabs(car.x - xPos2) - 10
            else:
                xDistance2 = 100

            if not railing.collide(car.ray2)[1] is None:
                yPos2 = car.ray2.y + railing.collide(car.ray2)[1][1]
                yDistance2 = math.fabs(car.y - yPos2) - 10
            else:
                yDistance2 = 100

            if not railing.collide(car.ray3)[1] is None:
                xPos3 = car.ray3.x + railing.collide(car.ray3)[1][0]
                xDistance3 = math.fabs(car.x - xPos3) - 10
            else:
                xDistance3 = 100

            if not railing.collide(car.ray3)[1] is None:
                yPos3 = car.ray3.y + railing.collide(car.ray3)[1][1]
                yDistance3 = math.fabs(car.y - yPos3) - 10
            else:
                yDistance3 = 100

            if not railing.collide(car.ray4)[1] is None:
                xPos4 = car.ray4.x + railing.collide(car.ray4)[1][0]
                xDistance4 = math.fabs(car.x - xPos4) - 10
            else:
                xDistance4 = 100

            if not railing.collide(car.ray4)[1] is None:
                yPos4 = car.ray4.y + railing.collide(car.ray4)[1][1]
                yDistance4 = math.fabs(car.y - yPos4) - 10
            else:
                yDistance4 = 100

            #Feed Input Nodes And Read Output
            output = nets[x].activate((xDistance, yDistance, xDistance1, yDistance1, xDistance2, yDistance2, xDistance3, yDistance3, xDistance4, yDistance4))

            #Check Output And According Turn
            if output[0] > 0.9:
                car.rotate(1)
            if output[0] < -0.9:
                car.rotate(-1)

            # Check For Collision
            if railing.collide(car)[0]:
                ge[x].fitness -= 1
                cars.pop(x)
                nets.pop(x)
                ge.pop(x)

        #Draw Game
        draw_window(win, cars, GEN)

def run(config_path):
    #Configure Config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    #Create Population
    p = neat.Population(config)

    #Run for up to 50 generations.
    winner = p.run(main, 1000)

    #Pickle Winner
    pickle.dump(winner, open("winner.pkl", "wb"))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
