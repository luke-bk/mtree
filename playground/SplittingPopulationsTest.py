from helpers.random_generator import RandomGenerator

rnd = RandomGenerator(seed=1)

for x in range(10):
    print(rnd.normal(0.0, 0.1))
print("---------------------------")
for x in range(10):
    print(rnd.normal(0.0, 0.01))
print("---------------------------")
for x in range(10):
    print(rnd.normal(0.0, 0.001))
