#include <bits/stdc++.h>
using namespace std;

class GraphToolkit {
    struct Edge {
        int u, v, weight;
        bool operator<(const Edge& other) const {
            return weight < other.weight;
        }
        Edge(int a, int b, int w = 0) {
            u = a;
            v = b;
            weight = w;
        }
    };

    struct Edge2 {
        int from, to, capacity, cost;
    };

   public:
    class DSU {
        std::vector<int> parent;
        std::vector<int> rank;

       public:
        DSU(int n) {
            parent.resize(n);
            rank.resize(n, 0);
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
        }
        int find(int x) {
            if (x != parent[x]) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
        void unionSets(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX != rootY) {
                if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
            }
        }
    };
    std::vector<Edge> kruskal(int n, std::vector<Edge>& edges) {
        std::sort(edges.begin(), edges.end());
        DSU dsu(n);
        std::vector<Edge> mst;
        for (const auto& edge : edges) {
            if (dsu.find(edge.u) != dsu.find(edge.v)) {
                dsu.unionSets(edge.u, edge.v);
                mst.push_back(edge);
            }
        }
        return mst;
    }

    vector<vector<int>> kosaraju(int n, vector<vector<int>> graph) {
        vector<vector<int>> reversedGraph(n + 1);
        vector<bool> visited(n + 1), visitedBackwards(n + 1);
        vector<int> comp(n + 1);
        stack<int> nodes;

        for (int i = 1; i <= n; ++i) {
            for (int x : graph[i]) {
                reversedGraph[x].push_back(i);
            }
        }

        function<void(int)> dfs1 = [&](int node) {
            visited[node] = true;
            for (int dest : graph[node]) {
                if (visited[dest])
                    continue;
                dfs1(dest);
            }
            nodes.push(node);
        };

        for (int i = 1; i <= n; ++i) {
            if (!visited[i]) {
                dfs1(i);
            }
        }

        vector<vector<int>> scc;
        vector<int> curr_scc;

        function<void(int)> dfs2 = [&](int node) {
            visitedBackwards[node] = true;
            comp[node] = scc.size();
            curr_scc.push_back(node);
            for (int dest : reversedGraph[node]) {
                if (visitedBackwards[dest])
                    continue;
                dfs2(dest);
            }
        };

        while (!nodes.empty()) {
            int curr_node = nodes.top();
            nodes.pop();
            if (visitedBackwards[curr_node]) {
                continue;
            }
            dfs2(curr_node);
            scc.push_back(curr_scc);
            curr_scc.clear();
        }
        return scc;
    }

    vector<pair<int, int>> get_bridges(int n, vector<vector<int>> graph) {
        vector<int> timeIn(n + 1), low(n + 1), seen(n + 1), root(n + 1),
            subtreeSz(n + 1);
        vector<pair<int, int>> bridges;
        int timer = 0, cnt_comp = 1;

        function<void(int, int, int)> traverse_graph = [&](int node, int comp,
                                                           int par) {
            timeIn[node] = low[node] = timer++;
            seen[node] = true;
            subtreeSz[node] = 1;
            for (int dest : graph[node]) {
                if (dest == par) {
                    continue;
                }
                if (timeIn[dest] == 0) {
                    traverse_graph(dest, comp, node);
                    subtreeSz[node] += subtreeSz[dest];
                    low[node] = min(low[node], low[dest]);
                    if (low[dest] > timeIn[node]) {
                        bridges.push_back({node, dest});
                    }
                } else {
                    low[node] = min(low[node], timeIn[dest]);
                }
            }
        };

        for (int i = 1; i <= n; ++i) {
            if (!seen[i]) {
                traverse_graph(i, cnt_comp, -1);
                cnt_comp++;
            }
        }
        return bridges;
    }

    vector<Edge> prim(int n, vector<vector<Edge>> graph) {
        std::vector<bool> inMST(n + 1, false);
        std::priority_queue<Edge> pq;
        int start = 1;
        inMST[start] = true;
        vector<Edge> MST;

        for (const auto& edge : graph[start]) {
            pq.push({edge.u, edge.v, -edge.weight});
        }
        while (!pq.empty()) {
            int node1 = pq.top().u, node2 = pq.top().v,
                currW = -pq.top().weight;
            pq.pop();
            if (inMST[node1] && inMST[node2]) {
                continue;
            }

            Edge e(node1, node2, currW);
            MST.push_back(e);

            inMST[node2] = true;
            for (const auto& edge : graph[node2]) {
                if (!inMST[edge.v]) {
                    pq.push({edge.u, edge.v, -edge.weight});
                }
            }
        }
        return MST;
    }

    vector<int> dijkstra(int V,
                         int start,
                         const vector<vector<pair<int, int>>>& graph) {
        priority_queue<pair<int, int>, vector<pair<int, int>>,
                       greater<pair<int, int>>>
            pq;
        vector<int> distances;

        distances.assign(V + 1, INT_MAX);
        distances[start] = 0;
        pq.push({0, start});

        while (!pq.empty()) {
            int currentDist = pq.top().first;
            int currentNode = pq.top().second;
            pq.pop();

            if (currentDist > distances[currentNode])
                continue;

            for (auto& neighbor : graph[currentNode]) {
                int nextNode = neighbor.first;
                int weight = neighbor.second;

                if (distances[currentNode] + weight < distances[nextNode]) {
                    distances[nextNode] = distances[currentNode] + weight;
                    pq.push({distances[nextNode], nextNode});
                }
            }
        }
        return distances;
    }

    void findArticulationPointsUtil(int node,
                                    int parent,
                                    vector<vector<int>>& adj,
                                    vector<bool>& visited,
                                    vector<int>& disc,
                                    vector<int>& low,
                                    vector<bool>& isArticulation,
                                    int& time) {
        visited[node] = true;
        disc[node] = low[node] = ++time;
        int children = 0;

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                children++;
                findArticulationPointsUtil(neighbor, node, adj, visited, disc,
                                           low, isArticulation, time);

                low[node] = min(low[node], low[neighbor]);

                if (parent == -1 && children > 1) {
                    isArticulation[node] = true;
                }
                if (parent != -1 && low[neighbor] >= disc[node]) {
                    isArticulation[node] = true;
                }
            } else if (neighbor != parent) {
                low[node] = min(low[node], disc[neighbor]);
            }
        }
    }
    vector<int> findArticulationPoints(int V, vector<vector<int>>& adj) {
        vector<bool> visited(V + 1, false);
        vector<int> disc(V + 1, -1);
        vector<int> low(V + 1, -1);
        vector<bool> isArticulation(V + 1, false);
        int time = 0;

        for (int i = 1; i <= V; i++) {
            if (!visited[i]) {
                findArticulationPointsUtil(i, -1, adj, visited, disc, low,
                                           isArticulation, time);
            }
        }

        vector<int> articulationPoints;
        for (int i = 1; i <= V; i++) {
            if (isArticulation[i]) {
                articulationPoints.push_back(i);
            }
        }

        return articulationPoints;
    }

    vector<int> topologicalSort(int V, vector<vector<int>>& adj) {
        vector<int> inDegree(V, 0);
        vector<int> topoOrder;
        queue<int> q;

        for (int i = 0; i < V; i++) {
            for (int neighbor : adj[i]) {
                inDegree[neighbor]++;
            }
        }

        for (int i = 0; i < V; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }

        while (!q.empty()) {
            int node = q.front();
            q.pop();
            topoOrder.push_back(node);

            for (int neighbor : adj[node]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
        return topoOrder;
    }

    vector<vector<int>> floydWarshall(int V, vector<vector<int>>& graph) {
        vector<vector<int>> dist = graph;
        for (int k = 0; k < V; k++) {
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
        return dist;
    }

    pair<bool, vector<int>> bellmanFord(int V, int src, vector<Edge>& edges) {
        vector<int> dist;
        dist.assign(V + 1, INT_MAX);
        dist[src] = 0;

        for (int i = 1; i <= V - 1; i++) {
            bool changed = false;
            for (auto [u, v, weight] : edges) {
                if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
        }

        for (auto [u, v, weight] : edges) {
            if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
                return {false, dist};
            }
        }

        return {true, dist};
    }

    int levenshteinDistance(string s, string t) {
        int const INF = 1e9;
        int n = s.size(), m = t.size();
        s = '.' + s;
        t = '.' + t;
        vector<vector<int>> minDist(n + 1, vector<int>(m + 1, INF));
        minDist[0][0] = 0;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= m; ++j) {
                if (i < n && j < m) {
                    minDist[i + 1][j + 1] =
                        min(minDist[i + 1][j + 1],
                            minDist[i][j] + (s[i + 1] != t[i + 1]));
                }
                if (j < m) {
                    minDist[i][j + 1] =
                        min(minDist[i][j + 1], minDist[i][j] + 1);
                }
                if (i < n) {
                    minDist[i + 1][j] =
                        min(minDist[i + 1][j], minDist[i][j] + 1);
                }
            }
        }
        return minDist[n][m];
    }

    int edmondsKarp(
        int n,
        int s,
        int t,
        vector<vector<int>> capacity,  // 0 for reversed, w for input edges
        vector<vector<int>> adj) {
        int flow = 0;
        int const INF = 1e9;
        vector<int> parent(n + 1);
        int new_tot_flow = 1;

        auto bfs = [&]() -> int {
            fill(parent.begin(), parent.end(), -1);
            parent[s] = -2;
            queue<pair<int, int>> q;
            q.push({s, INF});

            while (!q.empty()) {
                int cur = q.front().first;
                int curr_flow = q.front().second;
                q.pop();

                for (int next : adj[cur]) {
                    if (parent[next] == -1 && capacity[cur][next]) {
                        parent[next] = cur;
                        int new_flow = min(curr_flow, capacity[cur][next]);
                        if (next == t) {
                            return new_flow;
                        }
                        q.push({next, new_flow});
                    }
                }
            }

            return 0;
        };

        while ((new_tot_flow = bfs())) {
            flow += new_tot_flow;
            int cur = t;
            while (cur != s) {
                int prev = parent[cur];
                capacity[prev][cur] -= new_tot_flow;
                capacity[cur][prev] += new_tot_flow;
                cur = prev;
            }
        }

        return flow;
    }

    void six_colouring(int n, vector<vector<int>> graph, vector<int>& colour) {
        vector<int> deg(n + 1);
        vector<int> seen_colourings(n + 1);
        queue<int> buffer;
        stack<int> ready_nodes;

        for (int i = 1; i <= n; ++i) {
            deg[i] = graph[i].size();

            if (deg[i] <= 5) {
                buffer.push(i);
            }
        }

        while (!buffer.empty()) {
            int node = buffer.front();
            ready_nodes.push(node);
            buffer.pop();

            for (int dest : graph[node]) {
                deg[dest]--;
                if (deg[dest] == 5) {
                    buffer.push(dest);
                }
            }
        }

        while (!ready_nodes.empty()) {
            int node = ready_nodes.top();
            ready_nodes.pop();
            int bit = 0;
            while (seen_colourings[node] & (1 << bit)) {
                bit++;
            }
            colour[node] = bit + 1;
            for (int dest : graph[node]) {
                seen_colourings[dest] |= (1 << bit);
            }
        }
    }

    // nodes are indexed from 0 !!!
    // INF on cost when invalid edges

    int hamiltonian_cycle_with_minimum_cost(int n,
                                            vector<vector<int>> graph,
                                            vector<vector<int>> cost) {
        int const M = 1 << n, INF = 1e9;
        vector<vector<int>> min_cost(n,
                                     vector<int>(M, INF));  // cost[i][j][nodes]
        min_cost[0][1] = 0;

        // min_cost[i][i][1 << i] = 0;
        // min_cost[i][j][(1 << i) | (1 << j)] = min(min_cost[i][j][1 << i | 1
        // << j], min_cost[i][i][1 << i] + cost[i][j])

        for (int mask = 0; mask < M; ++mask) {
            for (int i = 0; i < n; ++i) {
                if (!(mask & (1 << i))) {
                    continue;
                }
                for (int dest : graph[i]) {
                    if (mask & (1 << dest)) {
                        continue;
                    }
                    int new_mask = mask | (1 << dest);
                    min_cost[dest][new_mask] =
                        min(min_cost[dest][new_mask],
                            min_cost[i][mask] + cost[i][dest]);
                }
            }
        }

        int min_cost_all = INF;
        for (int i = 0; i < n; ++i) {
            min_cost_all = min(min_cost[i][M - 1] + cost[i][0], min_cost_all);
        }

        return min_cost_all;
    }

    vector<int> get_eulerian_cycle(int n,
                                   int m,
                                   int start,
                                   vector<vector<pair<int, int>>> graph) {
        for (int i = 1; i <= n; ++i) {
            if (graph[i].size() % 2 == 1) {
                return {-1};
            }
        }

        vector<int> cycle;
        vector<bool> seen(m + 1);

        stack<int> nodes;
        nodes.push(start);

        while (!nodes.empty()) {
            int node = nodes.top();

            if (!graph[node].empty()) {
                int dest = graph[node].back().first;
                int num = graph[node].back().second;
                graph[node].pop_back();

                if (!seen[num]) {
                    seen[num] = true;
                    nodes.push(dest);
                }
            } else {
                nodes.pop();
                cycle.push_back(node);
            }
        }

        return cycle;
    }

    // len[i] - the length of the longest proper prefix that is also suffix for
    // s[0, i]
    void failure_function(string s, vector<int>& len) {
        int j = 0, i = 1, sz = s.size();
        while (i < sz) {
            if (s[i] == s[j]) {
                len[i++] = ++j;
            } else if (j == 0) {
                len[i++] = 0;
            } else {
                j = len[j - 1];
            }
        }
    }

    vector<int> zFunction(string s) {
        int l = 0, r = 0, idx = 1, sz = s.size();
        vector<int> maxLen(sz);
        while (idx < sz) {
            if (idx < r) {
                maxLen[idx] = min(r - idx, maxLen[idx - l]);
            }
            while (idx + maxLen[idx] < sz and
                   s[idx + maxLen[idx]] == s[maxLen[idx]]) {
                maxLen[idx]++;
            }
            if (idx + maxLen[idx] > r) {
                l = idx;
                r = idx + maxLen[idx];
            }
            idx++;
        }
        return maxLen;
    }
    // maxLen[i] = lungimea celui mai lung prefix pentru sirul s care incepe pe
    // pozitia i

    int min_cost_flow(int N, vector<Edge2> edges, int K, int s, int t) {
        vector<vector<int>> adj, cost, capacity;

        const int INF = 1e9;

        auto shortest_paths = [&](int n, int v0, vector<int>& d,
                                  vector<int>& p) {
            d.assign(n, INF);
            d[v0] = 0;
            vector<bool> inq(n, false);
            queue<int> q;
            q.push(v0);
            p.assign(n, -1);

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                inq[u] = false;
                for (int v : adj[u]) {
                    if (capacity[u][v] > 0 && d[v] > d[u] + cost[u][v]) {
                        d[v] = d[u] + cost[u][v];
                        p[v] = u;
                        if (!inq[v]) {
                            inq[v] = true;
                            q.push(v);
                        }
                    }
                }
            }
        };

        adj.assign(N, vector<int>());
        cost.assign(N, vector<int>(N, 0));
        capacity.assign(N, vector<int>(N, 0));
        for (Edge2 e : edges) {
            adj[e.from].push_back(e.to);
            adj[e.to].push_back(e.from);
            cost[e.from][e.to] = e.cost;
            cost[e.to][e.from] = -e.cost;
            capacity[e.from][e.to] = e.capacity;
        }

        int flow = 0;
        int min_cost = 0;
        vector<int> d, p;
        while (flow < K) {
            shortest_paths(N, s, d, p);
            if (d[t] == INF)
                break;

            int f = K - flow;
            int cur = t;
            while (cur != s) {
                f = min(f, capacity[p[cur]][cur]);
                cur = p[cur];
            }

            flow += f;
            min_cost += f * d[t];
            cur = t;
            while (cur != s) {
                capacity[p[cur]][cur] -= f;
                capacity[cur][p[cur]] += f;
                cur = p[cur];
            }
        }

        if (flow < K)
            return -1;
        else
            return min_cost;
    }
};

int main() {
    GraphToolkit toolkit;
    return 0;
}
