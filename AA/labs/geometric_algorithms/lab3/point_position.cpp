#include <bits/stdc++.h>
#define ll long long
#define ii pair<ll, ll>
using namespace std;


ll determinant4x4(vector<vector<ll>>& matrix) {
    ll det = 0;
    for (int col = 0; col < 4; ++col) {
        vector<vector<ll>> submatrix(3, vector<ll>(3));
        for (int i = 1; i < 4; ++i) {
            int subcol = 0;
            for (int j = 0; j < 4; ++j) {
                if (j == col) continue;
                submatrix[i-1][subcol] = matrix[i][j];
                ++subcol;
            }
        }

        ll subdet = 
            submatrix[0][0] * (submatrix[1][1] * submatrix[2][2] - submatrix[1][2] * submatrix[2][1]) -
            submatrix[0][1] * (submatrix[1][0] * submatrix[2][2] - submatrix[1][2] * submatrix[2][0]) +
            submatrix[0][2] * (submatrix[1][0] * submatrix[2][1] - submatrix[1][1] * submatrix[2][0]);
        det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * subdet;
    }
    return det;
}

ll get_point_positioning(vector<ii> points) {
    vector<vector<ll>> matrix(4, vector<ll>(4));

    for (int i = 0; i < 4; ++i) {
        matrix[i][0] = points[i].first;
        matrix[i][1] = points[i].second;
        matrix[i][2] = points[i].first * points[i].first + points[i].second * points[i].second;
        matrix[i][3] = 1;
    }

    return determinant4x4(matrix);
}

ll get_orientation_value(ii p1, ii p2, ii p3) {
    ll sum1 = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
    ll sum2 = p3.first * p2.second + p2.first * p1.second + p1.first * p3.second;
    return sum1 - sum2;
}

int main() {
    ios_base::sync_with_stdio(false); cin.tie(nullptr);
    vector<ii> points(4);
    for (int i = 0; i < 3; ++i) {
        cin >> points[i].first >> points[i].second;
    }

    if (get_orientation_value(points[0], points[1], points[2]) < 0) {
        swap(points[1], points[2]);
    }

    int queries;
    cin >> queries;
    while (queries--) {
        cin >> points[3].first >> points[3].second;

        ll val = get_point_positioning(points);

        if (val > 0) {
            cout << "INSIDE\n";
        } else if (val == 0) {
            cout << "BOUNDARY\n";
        } else {
            cout << "OUTSIDE\n";
        }
    }
    
    return 0;
}
