
import random
import matplotlib.pyplot as plt

class SmartphoneInventoryEnvironment:
    price_fluctuations = [0, 10, -20, 15, 0, -10, 5, -5, 20, 0]  # Example price changes
    noise = 5  # Random fluctuation in price

    def __init__(self):
        self.time = 0
        self.stock = 30  # Initial stock level
        self.price = 600  # Initial price of the smartphone
        self.price_history = [self.price]
        self.stock_history = [self.stock]

    def initial_percept(self):
        """Initial percept for the agent."""
        return {"price": self.price, "stock": self.stock}
def update_environment(self, order_quantity):
        """Simulate the environment after the agent's action."""
        self.stock += order_quantity
        self.stock -= random.randint(1, 6)  # Random customer purchases
        self.stock = max(self.stock, 0)  # Stock can't go negative
        self.time += 1
        # Update price based on fluctuations and random noise
        self.price += self.price_fluctuations[self.time % len(self.price_fluctuations)]
        self.price += random.gauss(0, self.noise)
        self.price = max(self.price, 100)  # Price can't go below 100
        self.price_history.append(self.price)
        self.stock_history.append(self.stock)
        return {"price": self.price, "stock": self.stock}
