import pygame_class
import pydoc

class pygame_class:
    """
    The pygame class is where the mini-game(s) is initialized, launched/run, and terminated.
    Each component of the class is designed to operate via their respecitve functions.

    Attributes:
    None

    Methods:
    init(): void
    startGameLoop(): void
    quit(): void
    """

    def init():
        """
        The respective pygame mini-game(s) is initialized, and the method is void returning no value.
        """
        return
    
    def startGameLoop():
        """
        Launches/runs the specified pygame mini-game, and the method is void returning no value.
        Pre-condition: init() has been called to initalize the game state.
        Post-condition: The game is currently being run.
        """
        return
    
    def quit():
        """
        This ends or terminates the game bringing the user back to the GUI or menu screen, and the method is void returning no value.
        Pre-condition: startGameLoop() is actively running.
        Post-condition: brings user back to menu screen.
        """
        return
    
pydoc.writedoc("pygame_class.py")