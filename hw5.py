# Exercise 1. Patient Class

# 1.1) Create a Patient class with attributes name and symptoms
        
class Patient:
    def __init__(self, name, symptoms):
        self.name = name
        self.symptoms = symptoms
        self.tests = {}
        self.covid_probability = 0.05

# 1.2) Method to add test results
    def add_test(self, test_name, test_result):
        self.tests[test_name] = test_result
        if test_name == "covid" and test_result:
            self.covid_probability = 0.99
        elif test_name == "covid" and not test_result:
            self.covid_probability = 0.01
        else:
            for symptom in ['fever', 'cough', 'anosmia']:
                if symptom in self.symptoms:
                    self.covid_probability += 0.1
                    
# 1.3) Method to check for the probability of having Covid
    def has_covid(self):
        return self.covid_probability

# Exercise 2. Card and Deck Class

import random

# 2.1) Create Card class with suit and value attributes
class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

# 2.2) Create Deck class with cards attribute
class Deck:
    def __init__(self):
        self.cards = []
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        for suit in suits:
            for value in values:
                self.cards.append(Card(suit, value))

# Method to shuffle the deck
    def shuffle(self): 
        random.shuffle(self.cards)

# Method to draw a card
    def draw(self):
        if len(self.cards) == 0:
            print("Deck is empty.")
        else:
            card = self.cards.pop()
            print(f"Suit: {card.suit}, Value: {card.value}")

# Exercise 3. PlaneFigure Class

from abc import ABC, abstractmethod
import math

# 3.1) Create abstract class PlaneFigure with two abstract methods
class PlaneFigure(ABC):
    @abstractmethod
    def compute_perimeter(self):
        pass
    
    @abstractmethod
    def compute_surface(self):
        pass

# 3.2) Create Triangle class inheriting from PlaneFigure
class Triangle(PlaneFigure):
    def __init__(self, base: float, c1: float, c2: float, height: float):
        self.base = base
        self.c1 = c1
        self.c2 = c2
        self.height = height
    
    def compute_perimeter(self):
        return self.base + self.c1 + self.c2
    
    def compute_surface(self):
        return 0.5 * self.base * self.height

# 3.3) Create Rectangle class inheriting from PlaneFigure
class Rectangle(PlaneFigure):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    
    def compute_perimeter(self):
        return 2 * (self.a + self.b)
    
    def compute_surface(self):
        return self.a * self.b

# 3.4) Create Circle class inheriting from PlaneFigure
class Circle(PlaneFigure):
    def __init__(self, radius: float):
        self.radius = radius
    
    def compute_perimeter(self):
        return 2 * math.pi * self.radius
    
    def compute_surface(self):
        return math.pi * (self.radius ** 2)