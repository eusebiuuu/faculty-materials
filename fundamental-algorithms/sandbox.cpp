#include <bits/stdc++.h>
#define ll long long
#define ii pair<int, int>
using namespace std;

struct Edge {
  int u, v, weight;
  bool operator<(const Edge& other) const { return weight < other.weight; }
  Edge(int a, int b, int w = 0) {
    u = a;
    v = b;
    weight = w;
  }
};

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  int n, m;
  cin >> n >> m;
  vector<vector<int>> graph(n + 1), capacity(n + 1, vector<int>(n + 1));
  for (int i = 1; i <= m; ++i) {
    int a, b, w;
    cin >> a >> b >> w;
    graph[a].push_back(b);
    graph[b].push_back(a);
    capacity[a][b] = w;
  }
  int s, t;
  cin >> s >> t;

  cout << '\n';
  return 0;
}
