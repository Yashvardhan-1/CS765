import json
from web3 import Web3
import argparse
import random
import numpy as np
from heapq import *  # For the event queues


#connect to the local ethereum blockchain
provider = Web3.HTTPProvider('http://127.0.0.1:8545')
w3 = Web3(provider)
#check if ethereum is connected
print(w3.is_connected())

#replace the address with your contract address (!very important)
deployed_contract_address = '0x4a960d01d977403b074e8fd2eff438474ed51e34'

#path of the contract json file. edit it with your contract json file
compiled_contract_path ="build/contracts/Payment.json"
with open(compiled_contract_path) as file:
    contract_json = json.load(file)
    contract_abi = contract_json['abi']
contract = w3.eth.contract(address = deployed_contract_address, abi = contract_abi)



'''
#Calling a contract function createAcc(uint,uint,uint)
txn_receipt = contract.functions.createAcc(1, 2, 5).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
txn_receipt_json = json.loads(w3.to_json(txn_receipt))
print(txn_receipt_json) # print transaction hash

# print block info that has the transaction)
print(w3.eth.get_transaction(txn_receipt_json)) 

#Call a read only contract function by replacing transact() with call()

'''

#Add your Code here
''' 
1) need to make a class to make a connected graph 
2) tranlate that graph into the solidity contract 
3) fire 1000 txns 
4) close connections 
'''

class Simulator:
    def __init__(self, args):
        self.n = args.n # Number of peers
        self.txns = args.txns
        self.lpeer = args.lpeer
        self.rpeer = args.rpeer
        self.peers = None
        self.network = None
        self.success = None

    def createNetwork(self):

        for i in self.n:
            pass
            # call register user 

        self.peers = [set() for i in range(self.n)]
        while True:
            self.network = set()
            for i in range(self.n):
                self.peers[i] = set() # Initialize neighbors to nullset
            
            for i in range(self.n):
                connections = random.randint(self.lpeer, self.rpeer) # Choose 3 to 6 neighbors

                for _ in range(connections):
                    connection = random.randint(0, self.n - 1) # Random neighbor id
                    if connection == i:
                        connection = (i+1)%self.n # Prevent self loop

                    self.peers[i].add(connection) # Both way connection
                    self.peers[connection].add(i)
                    self.network.add((i,connection))

            visited = set()
            queue = []
            start_node = 0
            queue.append(start_node)
            visited.add(start_node)
            #BFS to check if graph is connected
            
            while queue:
                node = queue.pop(0)
                for neighbor in self.peers[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(visited) == self.n:
                break

            # make a call to solidity functions to register and create account

            for node1, node2 in self.network:
                amount = np.exponential(scale=10, size=None)
                # amount = int(amount)+1;
                # call createAcc(node1, node2, amount/2)

                pass
    def run():
        success = 0

        for i in range(self.txns):
            if (i+1)%100:
                print(success/(i+1))

            s = random.randint(0, self.n - 1)
            e = random.randint(0, self.n - 1)
            if e == s:
                e = (s+1)%self.n
            # call solidity api 
            # sendAmount(s, e, 1)
            # return true on success+=1 

    def end():
        for n1, n2 in self.network:
            pass
            # closeAccount(n1, n2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default = 100)
    parser.add_argument("--txns", default = 1000)
    parser.add_argument("--lpeer", default = 3)
    parser.add_argument("--rpeer", default = 6)
    args = parser.parse_args()

    simulator = Simulator(args)
    simulator.createNetwork()
    simulator.run() 
    simulator.end() 
    
    
