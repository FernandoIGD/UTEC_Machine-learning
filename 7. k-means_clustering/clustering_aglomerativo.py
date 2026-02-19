import heapq
import math
import sys


def read_data():

    tokens = sys.stdin.buffer.read().split()
    if not tokens:
        raise ValueError("Input is empty.")
    if len(tokens) < 4:
        raise ValueError("Expected at least 4 values: N P K L.")

    n, p, k, l = map(int, tokens[:4])
    expected = 4 + n * p
    if len(tokens) != expected:
        raise ValueError(f"Expected {expected} total tokens, got {len(tokens)}.")

    coords = list(map(float, tokens[4:]))
    data = [coords[i * p:(i + 1) * p] for i in range(n)]
    return n, p, k, l, data


def euclidean_distance(point1, point2):
    """Compute Euclidean distance between two points.

    Args:
        point1 (list[float]): First point coordinates.
        point2 (list[float]): Second point coordinates.

    Returns:
        float: Euclidean distance between both points.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def pair_key(cluster_a, cluster_b):
    """Build an ordered key for a pair of cluster identifiers.

    Args:
        cluster_a (int): First cluster identifier.
        cluster_b (int): Second cluster identifier.

    Returns:
        tuple: Ordered tuple `(min_id, max_id)` for dictionary indexing.
    """
    if cluster_a < cluster_b:
        return cluster_a, cluster_b
    return cluster_b, cluster_a


def update_distance(linkage_type, distance_ax, distance_bx, size_a, size_b):
    """Update distance from a merged cluster to another cluster.

    Args:
        linkage_type (int): Linkage type (0=single, 1=complete, 2=average).
        distance_ax (float): Distance between cluster A and cluster X.
        distance_bx (float): Distance between cluster B and cluster X.
        size_a (int): Number of points in cluster A.
        size_b (int): Number of points in cluster B.

    Returns:
        float: Distance between merged cluster (A union B) and cluster X.
    """
    if linkage_type == 0:
        return min(distance_ax, distance_bx)
    if linkage_type == 1:
        return max(distance_ax, distance_bx)
    if linkage_type == 2:
        return (size_a * distance_ax + size_b * distance_bx) / (size_a + size_b)
    raise ValueError("Invalid linkage type. Use 0, 1, or 2.")


def aglomerative_clustering(data, k, linkage_type):
    """Run agglomerative clustering with deterministic tie-breaking.

    Args:
        data (list[list[float]]): Input points.
        k (int): Target number of final clusters.
        linkage_type (int): Linkage type (0=single, 1=complete, 2=average).

    Returns:
        tuple: A tuple `(merges, labels)` where:
            - `merges` (list[tuple]): Merge history tuples
              `(id_a, id_b, new_id, distance)`.
            - `labels` (list[int]): Final label for each original point index.
    """
    n = len(data)
    active = set(range(n))
    members = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    min_member = {i: i for i in range(n)}

    distances = {}
    heap = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(data[i], data[j])
            key = (i, j)
            distances[key] = dist
            heapq.heappush(heap, (dist, i, j))

    merges = []
    next_id = n

    while len(active) > k:
        while True:
            best_distance, id_a, id_b = heapq.heappop(heap)
            if id_a in active and id_b in active:
                current = distances.get(pair_key(id_a, id_b))
                if current is not None and best_distance == current:
                    break

        new_id = next_id
        next_id += 1

        merges.append((id_a, id_b, new_id, best_distance))

        active.remove(id_a)
        active.remove(id_b)
        active.add(new_id)

        members[new_id] = members[id_a] + members[id_b]
        sizes[new_id] = sizes[id_a] + sizes[id_b]
        min_member[new_id] = min(min_member[id_a], min_member[id_b])

        for other in list(active):
            if other == new_id:
                continue
            dist_a = distances[pair_key(id_a, other)]
            dist_b = distances[pair_key(id_b, other)]
            new_dist = update_distance(
                linkage_type, dist_a, dist_b, sizes[id_a], sizes[id_b]
            )
            key = pair_key(new_id, other)
            distances[key] = new_dist
            heapq.heappush(heap, (new_dist, key[0], key[1]))

    final_clusters = sorted(active, key=lambda cluster_id: min_member[cluster_id])
    cluster_to_label = {cluster_id: label for label, cluster_id in enumerate(final_clusters)}

    labels = [0] * n
    for cluster_id in final_clusters:
        label = cluster_to_label[cluster_id]
        for point_index in members[cluster_id]:
            labels[point_index] = label

    return merges, labels


def print_result(merges, labels):
    """Print clustering result in the required format.

    Args:
        merges (list[tuple]): Merge history `(id_a, id_b, new_id, distance)`.
        labels (list[int]): Final label per original point index.

    Returns:
        None: Writes formatted output to standard output.
    """
    for id_a, id_b, new_id, distance in merges:
        print(f"{id_a} {id_b} {new_id} {distance:.4f}")
    print(" ".join(map(str, labels)))


def main():
    """Execute the clustering program using stdin/stdout.

    Args:
        None: Entry point without arguments.

    Returns:
        None: Runs clustering and prints the result.
    """
    n, p, k, linkage_type, data = read_data()
    if not (1 <= k <= n <= 500):
        raise ValueError("Expected constraints: 1 <= K <= N <= 500.")
    if not (1 <= p <= 10):
        raise ValueError("Expected constraint: 1 <= P <= 10.")

    merges, labels = aglomerative_clustering(data, k, linkage_type)
    print_result(merges, labels)


if __name__ == "__main__":
    main()
