import matplotlib.pyplot as plt
import random
from controller.PriceMonitoringController import pricemonitoringcontroller
from controller.InventoryMonitoringController import inventorymonitorcontroller
from controller.OrderingController import decideorderquantitycontroller

class SmartphoneInventoryAgent:
    def __init__(self, average_price, min_order_quantity=10, discount_threshold=0.2, critical_stock_level=10):
        self.average_price = average_price
        self.min_order_quantity = min_order_quantity
        self.discount_threshold = discount_threshold
        self.critical_stock_level = critical_stock_level

    def decide_order(self, current_price, stock_level):
        return decideorderquantitycontroller(
            current_price=current_price,
            stock_level=stock_level,
            average_price=self.average_price,
            critical_stock_level=self.critical_stock_level,
            min_order_quantity=self.min_order_quantity,
            discount_threshold=self.discount_threshold
        )

# Run Simulation for Order Decisions
def run_simulation(agent, price_fluctuations, stock_levels):
    order_quantities = []

    for current_price, stock_level in zip(price_fluctuations, stock_levels):
        order_quantity = agent.decide_order(current_price, stock_level)
        order_quantities.append(order_quantity)

    return order_quantities

# Parameters
average_price = 600
agent = SmartphoneInventoryAgent(average_price=average_price)

# Generate random price fluctuations and stock levels
price_fluctuations = [random.randint(400, 800) for _ in range(8)]  # Random prices between 400 and 800
stock_levels = [random.randint(5, 25) for _ in range(8)]           # Random stock levels between 5 and 25

# Run simulation and capture order quantities
order_quantities = run_simulation(agent, price_fluctuations, stock_levels)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(price_fluctuations, label="Price", marker="o")
plt.plot(stock_levels, label="Stock Level", marker="x")
plt.plot(order_quantities, label="Order Quantity", marker="s", linestyle="--")

# Add numeric values next to each point
for i, (price, stock, order) in enumerate(zip(price_fluctuations, stock_levels, order_quantities)):
    plt.text(i, price, f"{price}", ha="center", va="bottom", fontsize=10, color="blue")  # Price
    plt.text(i, stock + 8, f"{stock}", ha="center", va="bottom", fontsize=10, color="red")  # Stock Level
    plt.text(i, order - 34, f"{order}", ha="center", va="bottom", fontsize=10, color="green")  # Order Quantity

plt.xlabel("Time Step")
plt.ylabel("Values")
plt.legend()
plt.title("Smartphone Inventory Management Simulation with Random Values")
plt.grid(True)
plt.show()
