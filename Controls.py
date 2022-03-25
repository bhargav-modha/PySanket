from json.tool import main
import pyautogui as pg
from scipy.fftpack import shift
import ctypes

def screenshot():
    im = pg.screenshot("myss.png")

def copy():
    pg.hotkey('ctrl','c')

def paste():
    pg.hotkey('ctrl','v')
    
def undo():
    pg.hotkey('ctrl','z')

def selectAll():
    pg.hotkey('ctrl','a')

def open():
    pg.hotkey('ctrl','o')

def cut():
    pg.hotkey('ctrl','x')

def save():
    pg.hotkey('ctrl','s')

def lockPc():
    ctypes.windll.user32.LockWorkStation()

def openStart():
    pg.hotkey('win')

def taskBar():
    pg.hotkey('ctrl','shift','esc')

def openSettings():
    pg.hotkey('win','i')