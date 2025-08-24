import random
from typing import List, Dict

class Solution:
    def generateNFT(self, config: dict, n: int) -> List[str]:
        res = []
        traits = config[traits]
        for _ in range(n):
            combination = {}
            for trait, values in traits.items():
                combination[trait] = random.choice(values)
            res.append(combination)


def test1():
    print(" ========= Test 1  =========")
    solution = Solution()
    config = {
        "name": "config-1",
        "size": "large",
        "traits": {
            "nose": ["pointy", "tiny", "flat"],
            "mouth": ["small", "wide", "thin"],
            "eyes": ["blue", "green", "brown"]
        }
    }
    result = solution.generateNFT(config, 5)
    printResult(result)

def test2():
    print(" ========= Test 2  =========")
    solution = Solution()
    config = {
        "name": "config-2",
        "size": "small",
        "traits": {
            "color": ["red", "blue", "green"],
            "shape": ["circle", "square"]
        }
    }
    result = solution.generateNFT(config, 3)
    printResult(result)

def test3():
    print(" ========= Test 3  =========")
    solution = Solution()
    config = {
        "name": "config-3",
        "size": "large",
        "traits": {
            "color": ["red", "blue", "green", "yellow", "purple"],
            "texture": ["smooth", "rough", "grainy"],
            "size": ["tiny", "small", "medium", "large"]
        }
    }
    result = solution.generateNFT(config, 3)
    printResult(result)

def printResult(result: List[str]):
    for each in result:
        print(each)

def main():
    test1()
    test2()
    test3()

if __name__ == "__main__":
    main()