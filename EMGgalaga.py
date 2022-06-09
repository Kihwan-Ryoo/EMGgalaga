# 2021 생체설계프로젝트 최종 코드
import sys
import numpy as np
import random
from time import sleep
import serial
import serial.tools.list_ports
import pygame
from pygame.locals import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtGui import QImage
import pyqtgraph as pg


def get_ports():
    ports = serial.tools.list_ports.comports()
    return ports


def findMSP430(portsFound):
    commPort = 'None'
    numConnection = len(portsFound)
    for i in range(0, numConnection):
        port = foundPorts[i]
        strPort = str(port)
        if 'Serial' in strPort:
            splitPort = strPort.split(' ')
            commPort = (splitPort[0])
    return commPort


foundPorts = get_ports()
connectPort =findMSP430(foundPorts)

if connectPort != 'None':
    ser = serial.Serial(port=connectPort,
                        baudrate=9600,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        timeout=1)
else:
    print('Connection Issue!')

print(ser.name)

group_L = 0
group_R = 0
group_L_text = ''
group_R_text = ''
checkScaler = 0
thr1, thr2, thr3, thr4 = 0, 0, 0, 0

# Game Variables
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 640

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (250, 250, 50)
RED = (250, 50, 50)

FPS = 60


class MyWindow(QMainWindow):
    def __init__(self, ser, surface, parent=None):
        super(MyWindow, self).__init__(parent)
        self.ser = ser

        self.setFixedWidth(1800)
        self.setFixedHeight(800)

        yheight = 2
        self.plot_widget1 = pg.PlotWidget()
        self.plot_widget1.scale(0.4, 1)
        self.plot_widget1.setBackground('w')
        self.plot_widget1.setYRange((-1) * yheight, yheight)
        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget2.scale(0.4, 1)
        self.plot_widget2.setBackground('w')
        self.plot_widget2.setYRange((-1) * yheight, yheight)
        self.plot_widget3 = pg.PlotWidget()
        self.plot_widget3.scale(0.4, 1)
        self.plot_widget3.setBackground('w')
        self.plot_widget3.setYRange((-1) * yheight, yheight)
        self.plot_widget4 = pg.PlotWidget()
        self.plot_widget4.scale(0.4, 1)
        self.plot_widget4.setBackground('w')
        self.plot_widget4.setYRange((-1) * yheight, yheight)

        self.x = np.array([0, 0])

        self.y1 = np.array([0, 0])
        pen1 = pg.mkPen(color=(255, 0, 0))
        self.data_line1 = self.plot_widget1.plot(self.x, self.y1, pen=pen1)
        self.y2 = np.array([0, 0])
        pen2 = pg.mkPen(color=(0, 0, 255))
        self.data_line2 = self.plot_widget2.plot(self.x, self.y2, pen=pen2)
        self.y3 = np.array([0, 0])
        pen3 = pg.mkPen(color=(0, 255, 0))
        self.data_line3 = self.plot_widget3.plot(self.x, self.y3, pen=pen3)
        self.y4 = np.array([0, 0])
        pen4 = pg.mkPen(color=(128, 64, 28))
        self.data_line4 = self.plot_widget4.plot(self.x, self.y4, pen=pen4)

        self.label1 = QLabel('Label for Left', self)
        self.label2 = QLabel('Label for Right', self)
        self.label3 = QLabel('None', self)
        self.label4 = QLabel('None', self)

        self.lcd = QSpinBox(self)
        self.lcd.setRange(-100, 100)
        self.lcd.valueChanged.connect(self.upd)

        self.surface = surface
        self.galaga = GameWidget(self.surface)

        self.mytimer = QTimer(self)
        self.mytimer.setInterval(10)
        self.mytimer.timeout.connect(self.update_plot_data)
        self.mytimer.start()

        widget = QWidget()

        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.plot_widget1)
        vbox_plot.addWidget(self.plot_widget2)
        vbox_plot.addWidget(self.plot_widget3)
        vbox_plot.addWidget(self.plot_widget4)

        hbox_label = QHBoxLayout()
        hbox_label.addStretch(5)
        hbox_label.addWidget(self.label1)
        hbox_label.addStretch(1)
        hbox_label.addWidget(self.label3)
        hbox_label.addStretch(5)
        hbox_label.addWidget(self.label2)
        hbox_label.addStretch(1)
        hbox_label.addWidget(self.label4)
        hbox_label.addStretch(5)

        vbox = QVBoxLayout()
        vbox.addLayout(vbox_plot, 10)
        vbox.addLayout(hbox_label, 1)

        hbox = QHBoxLayout(widget)
        hbox.addWidget(self.galaga, 2)
        hbox.addLayout(vbox, 5)

        self.setLayout(hbox)
        self.setCentralWidget(widget)

        self.setWindowTitle('EMG galaga')
        self.resize(1800, 640)
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def ReadValue(self):
        while ser.readable():
            data_buff1 = [0, 0, 0, 0, 0, 0]
            data_buff2 = [0, 0, 0, 0, 0, 0]
            Data = [0, 0, 0, 0]
            startbyte1 = self.ser.read(1)
            if (startbyte1 != b'') & (startbyte1 == b'\x81'):
                for i in range(6):
                    data_buff1[i] = self.ser.read(1)
                startbyte2 = self.ser.read(1)
                if (startbyte2 != b'') & (startbyte2 == b'\x81'):
                    for j in range(6):
                        data_buff2[j] = self.ser.read(1)
                    Data[0] = (int.from_bytes(data_buff1[0], byteorder='big') * 128) + int.from_bytes(data_buff1[1], byteorder='big') - 7000
                    Data[1] = (int.from_bytes(data_buff1[2], byteorder='big') * 128) + int.from_bytes(data_buff1[3], byteorder='big') - 7000
                    Data[2] = (int.from_bytes(data_buff2[0], byteorder='big') * 128) + int.from_bytes(data_buff2[1], byteorder='big') - 7000
                    Data[3] = (int.from_bytes(data_buff2[2], byteorder='big') * 128) + int.from_bytes(data_buff2[3], byteorder='big') - 7000
            return Data

    def update_plot_data(self):
        winsize = 200
        global group_L, group_R
        global checkScaler
        global thr1, thr2, thr3, thr4

        if len(self.x) >= winsize:
            self.x = self.x[1:]
        self.x = np.append(self.x, [self.x[-1] + 1])

        if (len(self.y1) >= winsize) and (len(self.y2) >= winsize) and (len(self.y3) >= winsize) and (len(self.y4) >= winsize):
            if checkScaler == 0:
                checkScaler = 1
                thr1 = abs(max(self.y1.min(), self.y1.max(), key=abs))
                thr2 = abs(max(self.y2.min(), self.y2.max(), key=abs))
                thr3 = abs(max(self.y3.min(), self.y3.max(), key=abs))
                thr4 = abs(max(self.y4.min(), self.y4.max(), key=abs))
            self.y1 = self.y1[1:]
            self.y2 = self.y2[1:]
            self.y3 = self.y3[1:]
            self.y4 = self.y4[1:]
        Data = self.ReadValue()
        self.lcd.setValue(Data[0])
        for a in range(4):
            Data[a] = Data[a] / 2000
        group_L, group_R = self.GetClass(Data)
        if group_L == 0:
            group_L_text = ' '
        elif group_L == 3:
            group_L_text = 'Flexion'
        elif group_L == 4:
            group_L_text = 'Extension'
        if group_R == 0:
            group_R_text = ' '
        elif group_R == 3:
            group_R_text = 'Flexion'
        elif group_R == 4:
            group_R_text = 'Extension'
        self.y1 = np.append(self.y1, [Data[0]])
        self.y2 = np.append(self.y2, [Data[1]])
        self.y3 = np.append(self.y3, [Data[2]])
        self.y4 = np.append(self.y4, [Data[3]])

        if (len(self.x) == len(self.y1)) and (len(self.x) == len(self.y2)) and (len(self.x) == len(self.y3)) and (len(self.x) == len(self.y4)):
            self.data_line1.setData(self.x, self.y1)  # L out
            self.data_line2.setData(self.x, self.y2)  # L in
            self.data_line3.setData(self.x, self.y3)  # R out
            self.data_line4.setData(self.x, self.y4)  # R in
            self.label3.setText(group_L_text)
            self.label4.setText(group_R_text)

    def GetClass(self, Data):
        global thr1, thr2, thr3, thr4
        global group_L, group_R
        y_pred3_L, y_pred4_L, y_pred3_R, y_pred4_R = 0, 0, 0, 0
        if (abs(Data[0]) >= (thr1 * 1.33)):
            y_pred4_L = 1
        if (abs(Data[1]) >= (thr2 * 1.45)):
            y_pred3_L = 1
        if (abs(Data[2]) >= (thr3 * 1.16)):
            y_pred4_R = 1
        if (abs(Data[3]) >= (thr4 * 1.45)):
            y_pred3_R = 1
        if (y_pred3_L == 1) and (y_pred4_L == 1):
            if (abs(Data[0]) > abs(Data[1])):
                group_L = 4
            elif (abs(Data[0]) <= abs(Data[1])):
                group_L = 3
        elif (y_pred3_L == 1) and (y_pred4_L == 0):
            group_L = 3 # Flexion
        elif (y_pred3_L == 0) and (y_pred4_L == 1):
            group_L = 4 # Extension
        elif (y_pred3_L == 0) and (y_pred4_L == 0):
            group_L = 0 # None
        if (y_pred3_L == 1) and (y_pred4_L == 1):
            if (abs(Data[2]) > abs(Data[3])):
                group_R = 4
            elif (abs(Data[2]) <= abs(Data[3])):
                group_R = 3
        elif (y_pred3_R == 1) and (y_pred4_R == 0):
            group_R = 3 # Flexion
        elif (y_pred3_R == 0) and (y_pred4_R == 1):
            group_R = 4 # Extension
        elif (y_pred3_R == 0) and (y_pred4_R == 0):
            group_R = 0 # Extension
        return group_L, group_R

    def upd(self):
        screen = pygame.display.get_surface()
        self.galaga.update(screen)
        self.update()


class Fighter(pygame.sprite.Sprite):
    def __init__(self):
        super(Fighter, self).__init__()
        self.image = pygame.image.load('./PyShooting/fighter.png')
        self.rect = self.image.get_rect()
        self.rect.x = int(WINDOW_WIDTH / 2)
        self.rect.y = WINDOW_HEIGHT - self.rect.height
        self.dx = 0
        self.dy = 0

    def update(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        if self.rect.x < 0 or self.rect.x + self.rect.width > WINDOW_WIDTH:
            self.rect.x -= self.dx

        if self.rect.y < 0 or self.rect.y + self.rect.height > WINDOW_HEIGHT:
            self.rect.y -= self.dy

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def collide(self, sprites):
        for sprite in sprites:
            if pygame.sprite.collide_rect(self, sprite):
                return sprite


class Missile(pygame.sprite.Sprite):
    def __init__(self, xpos, ypos, speed):
        super(Missile, self).__init__()
        self.image = pygame.image.load('./PyShooting/missile.png')
        self.rect = self.image.get_rect()
        self.rect.x = xpos
        self.rect.y = ypos
        self.speed = speed
        self.sound = pygame.mixer.Sound('./PyShooting/missile.wav')

    def launch(self):
        self.sound.play()

    def update(self):
        self.rect.y -= self.speed
        if self.rect.y + self.rect.height < 0:
            self.kill()

    def collide(self, sprites):
        for sprite in sprites:
            if pygame.sprite.collide_rect(self, sprite):
                return sprite


class Rock(pygame.sprite.Sprite):
    def __init__(self, xpos, ypos, speed):
        super(Rock, self).__init__()
        rock_images = ('./PyShooting/rock01.png', './PyShooting/rock02.png', './PyShooting/rock03.png', './PyShooting/rock04.png', './PyShooting/rock05.png', \
                       './PyShooting/rock06.png', './PyShooting/rock07.png', './PyShooting/rock08.png', './PyShooting/rock09.png', './PyShooting/rock10.png', \
                       './PyShooting/rock11.png', './PyShooting/rock12.png', './PyShooting/rock13.png', './PyShooting/rock14.png', './PyShooting/rock15.png', \
                       './PyShooting/rock16.png', './PyShooting/rock17.png', './PyShooting/rock18.png', './PyShooting/rock19.png', './PyShooting/rock20.png', \
                       './PyShooting/rock21.png', './PyShooting/rock22.png', './PyShooting/rock23.png', './PyShooting/rock24.png', './PyShooting/rock25.png', \
                       './PyShooting/rock26.png', './PyShooting/rock27.png', './PyShooting/rock28.png', './PyShooting/rock29.png', './PyShooting/rock30.png')
        self.image = pygame.image.load(random.choice(rock_images))
        self.rect = self.image.get_rect()
        self.rect.x = xpos
        self.rect.y = ypos
        self.speed = speed

    def update(self):
        self.rect.y += self.speed

    def out_of_screen(self):
        if self.rect.y > WINDOW_HEIGHT:
            return True


def draw_text(text, font, surface, x, y, main_color):
    text_obj = font.render(text, True, main_color)
    text_rect = text_obj.get_rect()
    text_rect.centerx = x
    text_rect.centery = y
    surface.blit(text_obj, text_rect)


def occur_explosion(surface, x, y):
    explosion_image = pygame.image.load('./PyShooting/explosion.png')
    explosion_rect = explosion_image.get_rect()
    explosion_rect.x = x
    explosion_rect.y = y
    surface.blit(explosion_image, explosion_rect)

    explosion_sounds = ('./PyShooting/explosion01.wav', './PyShooting/explosion02.wav', './PyShooting/explosion03.wav')
    explosion_sound = pygame.mixer.Sound(random.choice(explosion_sounds))
    explosion_sound.play()


def game_loop():
    global group_L
    global group_R
    default_font = pygame.font.Font('./PyShooting/NanumGothic.ttf', 28)
    background_image = pygame.image.load('./PyShooting/background.png')
    gameover_sound = pygame.mixer.Sound('./PyShooting/gameover.wav')
    pygame.mixer.music.load('./PyShooting/music.wav')
    pygame.mixer.music.play(-1)
    fps_clock = pygame.time.Clock()

    fighter = Fighter()
    missiles = pygame.sprite.Group()
    rocks = pygame.sprite.Group()

    occur_prob = 200
    shot_count = 0
    count_missed = 0

    done = False

    while not done:

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    fighter.dx -= 5
                elif event.key == pygame.K_RIGHT:
                    fighter.dx += 5
                elif event.key == pygame.K_UP:
                    fighter.dy -= 5
                elif event.key == pygame.K_DOWN:
                    fighter.dy += 5
                elif event.key == pygame.K_SPACE:
                    missile = Missile(fighter.rect.centerx, fighter.rect.y, 10)
                    missile.launch()
                    missiles.add(missile)
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    fighter.dx = 0
                elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    fighter.dy = 0

        if (group_L, group_R) == (4, 4):
            fighter.dy += 1
        elif (group_L, group_R) == (3, 3):
            fighter.dy -= 1
        elif (group_L, group_R) == (4, 0):
            fighter.dx -= 1
        elif (group_L, group_R) == (0, 4):
            fighter.dx += 1

        test_mis = random.randint(0, 20)

        if test_mis == 1:
            missile = Missile(fighter.rect.centerx, fighter.rect.y, 10)
            missile.launch()
            missiles.add(missile)

        screen.blit(background_image, background_image.get_rect())

        occur_of_rocks = 1 + int(shot_count / 300)
        min_rock_speed = 1 + int(shot_count / 200)
        max_rock_speed = 1 + int(shot_count / 100)

        if random.randint(1, occur_prob) == 1:
            for i in range(occur_of_rocks):
                speed = random.randint(min_rock_speed, max_rock_speed)
                rock = Rock(random.randint(0, WINDOW_WIDTH - 30), 0, speed)
                rocks.add(rock)

        draw_text('파괴한 운석 : {}'.format(shot_count), default_font, screen, 100, 20, YELLOW)
        draw_text('놓친 운석 : {}'.format(count_missed), default_font, screen, 400, 20, RED)

        for missile in missiles:
            rock = missile.collide(rocks)
            if rock:
                missile.kill()
                rock.kill()
                occur_explosion(screen, rock.rect.x, rock.rect.y)
                shot_count += 1

        for rock in rocks:
            if rock.out_of_screen():
                rock.kill()
                count_missed += 1

        rocks.update()
        rocks.draw(screen)
        missiles.update()
        missiles.draw(screen)
        fighter.update()
        fighter.draw(screen)
        pygame.display.flip()

        if fighter.collide(rocks) or count_missed >= 3:
            pygame.mixer_music.stop()
            occur_explosion(screen, fighter.rect.x, fighter.rect.y)
            pygame.display.update()
            gameover_sound.play()
            sleep(1)
            done = True

        fps_clock.tick(FPS)

    return 'game_menu'


def game_menu():
    start_image = pygame.image.load('./PyShooting/background.png')
    screen.blit(start_image, [0, 0])
    draw_x = int(WINDOW_WIDTH / 2)
    draw_y = int(WINDOW_HEIGHT / 4)
    font_70 = pygame.font.Font('./PyShooting/NanumGothic.ttf', 70)
    font_40 = pygame.font.Font('./PyShooting/NanumGothic.ttf', 40)

    draw_text('지구를 지켜라!', font_70, screen, draw_x, draw_y, YELLOW)
    draw_text('엔터 키를 누르면', font_40, screen, draw_x, draw_y + 200, WHITE)
    draw_text('게임이 시작됩니다.', font_40, screen, draw_x, draw_y + 250, WHITE)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                return 'play'
        if event.type == QUIT:
            return 'quit'

    return 'game_menu'


def main():
    global screen

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), vsync=1)
    pygame.display.set_caption('PyShooting')

    app = QApplication(sys.argv)
    window = MyWindow(ser, screen)
    window.show()

    action = 'game_menu'
    while action != 'quit':
        if action == 'game_menu':
            action = game_menu()
        elif action == 'play':
            action = game_loop()

    pygame.quit()
    sys.exit(app.exec_())


class GameWidget(QWidget, QThread):
    def __init__(self, surface, parent=None):
        super(GameWidget, self).__init__(parent)
        self.w = surface.get_width()
        self.h = surface.get_height()
        self.data = surface.get_buffer().raw
        self.image = QtGui.QImage(self.data, self.w, self.h, QtGui.QImage.Format_RGB32)

    def update(self, surface):
        surface = pygame.display.get_surface()
        self.data = surface.get_buffer().raw
        self.image = QImage(self.data, self.w, self.h, QImage.Format_RGB32)
        pygame.display.flip()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.drawImage(0, 0, self.image)
        qp.end()


if __name__ == '__main__':
    main()