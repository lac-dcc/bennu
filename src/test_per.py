from itertools import permutations

def permutation(arr, limit):
    result = []
    for i, perm in enumerate(permutations(arr, len(arr))):
        if i >= limit:
            break
        result.append(list(perm))
    return result