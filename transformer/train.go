package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"
)

func readFile(fileName string) (string, error) {
	// Read the file
	content, err := ioutil.ReadFile(fileName)
	if err != nil {
		fmt.Println(err)
	}
	return string(content), nil
}

func mapCharToIdx(vocab string) map[string]int {
	charToIdx := map[string]int{}
	for i, char := range vocab {
		charToIdx[string(char)] = i
	}
	return charToIdx
}

func mapIdxToChar(vocab string) map[int]string {
	idxToChar := map[int]string{}
	for i, char := range vocab {
		idxToChar[i] = string(char)
	}
	return idxToChar
}

func encodeString(text string, charToIdx map[string]int) []int {
	encodedText := []int{}
	for _, char := range text {
		encodedText = append(encodedText, charToIdx[string(char)])
	}
	return encodedText
}

func decodeString(encodedText []int, idxToChar map[int]string) string {
	decodedText := ""
	for _, idx := range encodedText {
		decodedText += idxToChar[idx]
	}
	return decodedText
}

func main() {
	text, err := readFile("data.txt")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// print the length of dataset in characters
	// fmt.Println("Length of text: ", len(text))
	// substring := text[:1000]
	// fmt.Println("Substring: ", substring)


	// sets do not exist in golang, so we will use a map for the characters in text and it will behave as a set
	vocabSet := map[string]struct{}{}

	// iterate over the text and add each character to the set
	for _, char := range text {
		vocabSet[string(char)] = struct{}{}
	}

	// print the number of unique characters in the text
	fmt.Println("Number of unique characters: ", len(vocabSet))

	// create a list of unique characters
	var vocabSlice []string
	for char := range vocabSet {
		vocabSlice = append(vocabSlice, char)
	}

	sort.Strings(vocabSlice)

	sortedVocabStr := strings.Join(vocabSlice, "")
	fmt.Println("Sorted Vocab: ", sortedVocabStr)



	// 	TOKENIZATION!!!
	// create a mapping from character to index and index to character
	stoi := mapCharToIdx(sortedVocabStr)
	itos := mapIdxToChar(sortedVocabStr)
	
	// testing encoding and decoding of text
	encodedText := encodeString("hii there", stoi)
	decodedText := decodeString(encodedText, itos)

	fmt.Println("Encoded Text: ", encodedText)
	fmt.Println("Decoded Text: ", decodedText)

}