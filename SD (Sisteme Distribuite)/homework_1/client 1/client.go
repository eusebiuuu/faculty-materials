package main

import (
	"bytes"
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"errors"
	"strconv"
	"time"
	"os"
)

type RequestPayload1 struct {
	Words  []string `json:"words"`
	Type   int      `json:"type"`
}

type ResponsePayload1 struct {
	Words  []string `json:"words"`
}

var BASE_URL string = "http://localhost:8080/";

func exercise1() (RequestPayload1, string, error) {
	filename := "/home/eusebiuu/Documents/software_projects/distributed_systems/homework_1/client 1/input.txt"

	file, err := os.Open(filename)

	if err != nil {
		return RequestPayload1{}, "", fmt.Errorf("error opening file %s: %w", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file) 

	var count int
	var size int = -1

	scanner.Scan()
	firstLine := scanner.Text()
	count, err = strconv.Atoi(firstLine)

	url := fmt.Sprintf("%s1", BASE_URL)

	var words = make([]string, 0, count)
	for i := 0; i < count; i++ {
		scanner.Scan()
		var word = scanner.Text()
		words = append(words, word)

		if size == -1 {
			size = len(word)
			continue
		}
		if len(word) != size {
			return RequestPayload1{}, "", errors.New("All words must have the same size")
		}
	}

	data := RequestPayload1{
		Words: words,
		Type: 1,
	}

	return data, url, nil
}

func main() {
	fmt.Println("The client 1 has connected")
	var bodyReader *bytes.Buffer

	data, url, err := exercise1()
	jsonPayload, err := json.Marshal(data)
	if err != nil {
		log.Fatalf("Error marshaling payload: %v", err)
	}
	bodyReader = bytes.NewBuffer(jsonPayload)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bodyReader)
	if err != nil {
		log.Fatalf("Error creating request: %v", err)
	}
	
	// 4. Set the Content-Type header
	req.Header.Set("Content-Type", "application/json")
	
	// Execute the request
	fmt.Printf("The client is making the following request: %s\n", string(jsonPayload))
	resp, err := http.DefaultClient.Do(req)

	if err != nil {
		log.Fatalf("Error sending request: %v", err)
	}
	defer resp.Body.Close()

	// Read and print the server's response
	responseBody, _ := io.ReadAll(resp.Body)
	// fmt.Printf("Server Status: %s\n", resp.Status)
	// fmt.Printf("Server Response Body: %s\n", string(responseBody))
	fmt.Printf("The client got the response from the server: %s\n", string(responseBody))
}