"""Script to illustrate an iterative way (DP) to calculate levenshtein distance"""


def iterative_levenshtein(s, t, **weight_dict):  # 会把所有（不定数量）的输入转为一个dict
    """
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t

        weight_dict: keyword parameters setting the costs for characters,
                     the default value for a character will be 1
    """
    rows = len(s) + 1
    cols = len(t) + 1

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    w = dict((x, (1, 1, 1)) for x in alphabet + alphabet.upper())
    if weight_dict:
        w.update(weight_dict)

    dist = [[0 for x in range(cols)] for x in range(rows)]
    # source prefixes can be transformed into empty strings
    # by deletions:
    for row in range(1, rows):
        dist[row][0] = dist[row - 1][0] + w[s[row - 1]][0]
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for col in range(1, cols):
        dist[0][col] = dist[0][col - 1] + w[t[col - 1]][1]

    for col in range(1, cols):
        for row in range(1, rows):
            deletes = w[s[row - 1]][0]
            inserts = w[t[col - 1]][1]
            subs = max((w[s[row - 1]][2], w[t[col - 1]][2]))
            if s[row - 1] == t[col - 1]:
                subs = 0
            else:
                subs = subs
            dist[row][col] = min(dist[row - 1][col] + deletes,
                                 dist[row][col - 1] + inserts,
                                 dist[row - 1][col - 1] + subs)  # substitution
    for r in range(rows):  # print matrix
        print(dist[r])

    return dist[row][col]


if __name__ == '__main__':
    # default:
    print(iterative_levenshtein("abx",
                                "xya",
                                x=(3, 2, 8),
                                y=(4, 5, 4),
                                a=(7, 6, 6)))

    # print(iterative_levenshtein("abc", "xyz", costs=(1,1,substitution_costs)))
