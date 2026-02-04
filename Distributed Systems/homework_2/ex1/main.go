package main

import (
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"
)

type Pair [2]int

var VOWELS = "aeiou"

func validWord(word string) bool {
	return strings.Contains(VOWELS, string(word[0])) && strings.Contains(VOWELS, string(word[len(word)-1]))
}

func Map(in [][]string) []Pair {
	result := make([]Pair, 0)
	
	for _, wordsList := range in {
		var count = 0
		for _, word := range wordsList {
			if validWord(word) {
				count++
			}
		}
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

// Server function to handle map requests
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

// Client function to send map requests
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
		{"ana", "parc", "impare", "era", "copil"},
		{"cer", "program", "leu", "alee", "golang", "info"},
		{"inima", "impar", "apa", "eleve"},
	}

	serverAddress := "localhost:8080"

	// Start server in a separate goroutine
	go server(serverAddress, input)

	// Allow server to start
	time.Sleep(500 * time.Millisecond)

	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()

		// Get map results from server via client
		clientResults := client(serverAddress)
		if clientResults == nil {
			return
		}

		// Calculate average using Reduce
		result := Reduce(clientResults)
		fmt.Printf("Final answer is %f\n", result)
	}()
	wg.Wait()
	
}
