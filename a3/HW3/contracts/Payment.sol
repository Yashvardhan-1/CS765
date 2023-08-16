// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract Payment {
  uint numUsers;
  mapping (uint => string) public userNames;
  struct Edge {
    uint dest;
    uint bal;
  }

  mapping(uint => Edge[]) private adjList;

  constructor() {
    numUsers = 0;
  }

  function registerUser(uint user_id, string memory user_name) public returns (bool){
    if(bytes(userNames[user_id]).length > 0){
      return false;
    }
    userNames[user_id] = user_name;
    numUsers+=1;
    return true;
  } 

  function createAcc(uint user_id_1, uint user_id_2, uint init_balance) public {
    adjList[user_id_1].push(Edge({dest: user_id_1, bal: init_balance}));
    adjList[user_id_2].push(Edge({dest: user_id_1, bal: init_balance}));
  }

  function sendAmount(uint from, uint to, uint ammount) public returns (bool) {
    uint maxD = 10000;
    bool[] memory visited = new bool[](numUsers);
    uint[] memory queue = new uint[](numUsers);
    uint[] memory dist = new uint[](numUsers);
    uint[] memory prev = new uint[](numUsers);
    uint numQueue;
    bool found = false;

    for(uint i=0; i<numUsers; i++){
      dist[i] = maxD;
      visited[i] = false;
    }

    dist[from] = 0;
    visited[from] = true;
    queue[0] = from;
    numQueue = 1; 

    while(!found && numQueue>0){
      uint tnumQueue = 0;
      uint[] memory tqueue = new uint[](numUsers);
      for(uint i=0; i<numQueue; i++){
        uint s = queue[i];
        if(found){
          break;
        }
        for(uint j=0; j<adjList[s].length; j++){
          if(!visited[adjList[s][j].dest]){
          // if(!visited[adjList[s][j].dest]  && adjList[s][j].bal >= ammount){
            prev[adjList[s][j].dest] = s;
            tqueue[tnumQueue] = adjList[s][j].dest;
            dist[adjList[s][j].dest] = dist[s]+1;
            tnumQueue+=1;
          }
          if(adjList[s][j].dest == to){
            found = true;
            break;
          }
          visited[adjList[s][j].dest] = true;
        }
      }
      queue = tqueue;
      numQueue = tnumQueue;
    }


    uint node = to;
    uint pnode;

    while(node != from && found){
      pnode = prev[node];
      for(uint i=0; i < adjList[pnode].length; i++){
        if(adjList[pnode][i].dest == node){
          if(adjList[pnode][i].bal<ammount){
            found = false;
          }
          break;
        }
      }
    }

    if(!found) return false;
    
    node = to;
    pnode;

    while(node != from){
      pnode = prev[node];
      for(uint i=0; i < adjList[pnode].length; i++){
        if(adjList[pnode][i].dest == node){
          adjList[pnode][i].bal -= ammount;
          break;
        }
      }
      for(uint i=0; i < adjList[node].length; i++){
        if(adjList[node][i].dest == pnode){
          adjList[node][i].bal += ammount;
          break;
        }
      }
      node = pnode;
    }
    return true;
  }

  function closeAccount(uint user_id_1, uint user_id_2) public returns(bool){
    if(bytes(userNames[user_id_1]).length == 0){
      return false;
    }
    if(bytes(userNames[user_id_2]).length == 0){
      return false;
    }
    for(uint i=0; i<adjList[user_id_1].length; i++){
      if(adjList[user_id_1][i].dest == user_id_2){
        adjList[user_id_1][i] = adjList[user_id_1][adjList[user_id_1].length-1];
        adjList[user_id_1].pop();
        break;
      }
      if(i == adjList[user_id_1].length-1) return false;
    }
    for(uint i=0; i<adjList[user_id_2].length; i++){
      if(adjList[user_id_2][i].dest == user_id_1){
        adjList[user_id_2][i] = adjList[user_id_2][adjList[user_id_2].length-1];
        adjList[user_id_2].pop();
        break;
      }
      if(i == adjList[user_id_2].length-1) return false;
    }
    return true;
  }
}

  // function dijkstra(uint256 start, uint256 end, uint256 minWeight) internal returns (uint256[] memory, int256) {
  //     int[] memory distances = new int[](numUsers);
  //     uint[] memory previous = new uint[](numUsers);
  //     bool[] memory visited = new bool[](numUsers);

  //     for (uint256 i = 0; i < numUsers; i++) {
  //       distances[i] = -1;
  //       visited[i] = false;
  //     }

  //     distances[uint256(start)] = 0;

  //     while (true) {
  //       int256 current = -1;
  //       int256 minDist = -1;

  //       for (uint256 i = 0; i < uint256(numUsers); i++) {
  //           if (!visited[i] && distances[uint256(i)] < minDist) {
  //               current = int256(i);
  //               minDist = distances[uint256(i)];
  //           }
  //       }

  //       if (current == -1) {
  //           break;
  //       }

  //       visited[uint(current)] = true;

  //       for (uint256 j = 0; j < adjacencyList[uint(current)].length; j++) {
  //         uint256 neighbor = adjacencyList[uint(current)][j].dest;
  //         uint256 weight = adjacencyList[uint(current)][j].weight;

  //         if (weight >= minWeight) {
  //           uint256 alt = uint256(distances[uint(current)]) + weight;

  //           if (alt < uint256(distances[neighbor])) {
  //               distances[neighbor] = int256(alt);
  //               previous[neighbor] = uint256(current);
  //           }
  //         }
  //       }
  //     }

  //     uint256[] memory path = new uint256[](numUsers);
  //     uint256 pathLength = 0;

  //     uint256 node = end;
  //     uint256 t = end;

  //     while (node != start) {
  //       path[pathLength] = node;
  //       pathLength++;
  //       t = previous[node];

  //       // need to make transaction here 

  //       node = t;
  //     }

  //     path[pathLength] = start;
  //     pathLength++;

  //     for (uint256 i = 0; i < pathLength / 2; i++) {
  //       uint256 temp = path[i];
  //       path[i] = path[pathLength - i - 1];
  //       path[pathLength - i - 1] = temp;
  //     }

  //     return (path, distances[end]);
  //   }
