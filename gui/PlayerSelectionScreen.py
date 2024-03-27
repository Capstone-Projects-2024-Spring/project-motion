import pygame
import pydoc
class PlayerSelectionScreen:
    
    """
    PlayerSelectionScreen class manages the player selection interface of the application.
    It displays the number of player options and handles user selection.

    Attributes:
        numOfPlayers (int): The number of player options to display.
    """

    def __init__(self, numOfPlayers:int):
        """
        Initializes the PlayerSelectionScreen instance with the number of players.

        Parameters:
            numOfPlayers (int): number of players being used in the game.
        """
        self.numOfPlayers = numOfPlayers  

    def displayOptions(self):
        
        """
        Displays the player options (single player or multiplayer) based on the number of players.
        No pre-conditions.
        Post-conditions: Player options are displayed to the user.
        """  

        return

    def handleSelection(selection:int):
        """
        Determines the game's mode (single player or multiplayer) based off of the number of players.

        Pre-conditions:
            displayOptions has been called, and user input has been received.
        Post-conditions:
            The action corresponding to the user's selection is executed.
        Parameters:
            selection (int): The user's selection of the number of players.
            
        Returns:
            Nothing
            
        Raises:
            ValueError: If the selection is not within the valid range of player options.
        """
        return 

    pydoc.writedoc("PlayerSelectionScreen") 
