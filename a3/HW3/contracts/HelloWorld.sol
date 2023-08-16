// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract HelloWorld {
  constructor() public {
  }

  // Data location must be "memory" or "calldata" for return parameter in function, but none was given.
  function hi() public pure returns (string memory) {
    return ("Hello World");
  }

}
