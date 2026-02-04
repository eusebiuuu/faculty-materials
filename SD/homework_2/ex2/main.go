package main

import (
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"
)

type Pair [2]int

func reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < len(r)/2; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

func isPalindrome(word string) bool {
	return word == reverse(word)
}

func Map(in [][]string) []Pair {
	result := make([]Pair, 0)
	
	for _, wordsList := range in {
		var count = 0
		for _, word := range wordsList {
			if isPalindrome(word) {
				count++
			}
			// fmt.Println(word)
		}
		// fmt.Println(count)
		result = append(result, Pair{count, len(wordsList)})
	}
	return result
}

func Reduce(in []Pair) float64 {
	totalCount := 0
	sum := 0

	for _, kv := range in {
		sum += kv[0]
		totalCount++
	}

	return float64(sum) / float64(totalCount)
}

func server(address string, input [][]string) {
	ln, err := net.Listen("tcp", address)
	if err != nil {
		fmt.Println("Error starting server:", err)
		return
	}
	defer ln.Close()
	fmt.Println("Server is running on", address)

	for {
		conn, err := ln.Accept()
		if err != nil {
			fmt.Println("Connection error:", err)
			continue
		}
		go handleConnection(conn, input)
	}
}

func handleConnection(conn net.Conn, input [][]string) {
	defer conn.Close()

	mapResults := Map(input)
	data, err := json.Marshal(mapResults)
	if err != nil {
		fmt.Println("Error marshaling data:", err)
		return
	}
	conn.Write(data)
}

func client(address string) []Pair {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		fmt.Println("Error connecting to server:", err)
		return nil
	}
	defer conn.Close()

	buffer := make([]byte, 4096)
	n, err := conn.Read(buffer)
	if err != nil {
		fmt.Println("Error reading data from server:", err)
		return nil
	}

	var mapResults []Pair
	err = json.Unmarshal(buffer[:n], &mapResults)
	if err != nil {
		fmt.Println("Error unmarshaling data:", err)
		return nil
	}

	return mapResults
}

func main() {
	input := [][]string{
		{"a1551a", "parc", "ana", "minim", "1pcl3"},
		{"calabalac", "tivit", "leu", "zece10", "ploaie","9ana9"},
		{"lalalal", "tema", "papa", "ger"},
	}

	serverAddress := "localhost:8080"

	go server(serverAddress, input)

	time.Sleep(500 * time.Millisecond)

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()

		clientResults := client(serverAddress)
		if clientResults == nil {
			return
		}

		result := Reduce(clientResults)
		fmt.Printf("Final answer is %f\n", result)
	}()
	wg.Wait()
	
}
