package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/exp/rand"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func readFile(fileName string) (string, error) {
	content, err := ioutil.ReadFile(fileName)
	if err != nil {
		return "", err
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

func get_batch(split string, train_data, val_data []int, batch_size, block_size int) (*tensor.Dense, *tensor.Dense) {
	var data []int
	if split == "train" {
		data = train_data
	} else {
		data = val_data
	}

	rand.Seed(uint64(time.Now().UnixNano()))

	ix := make([]int, batch_size)
	for i := range ix {
		ix[i] = rand.Intn(len(data) - block_size)
	}

	xData := make([]int, batch_size*block_size)
	yData := make([]int, batch_size*block_size)

	for i := 0; i < batch_size; i++ {
		for j := 0; j < block_size; j++ {
			xData[i*block_size+j] = data[ix[i]+j]
			yData[i*block_size+j] = data[ix[i]+j+1]
		}
	}

	x := tensor.New(tensor.WithShape(batch_size, block_size), tensor.Of(tensor.Int), tensor.WithBacking(xData))
	y := tensor.New(tensor.WithShape(batch_size, block_size), tensor.Of(tensor.Int), tensor.WithBacking(yData))

	return x, y
}

func main() {
	text, err := readFile("data.txt")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	vocabSet := map[string]struct{}{}
	for _, char := range text {
		vocabSet[string(char)] = struct{}{}
	}

	fmt.Println("Number of unique characters: ", len(vocabSet))

	var vocabSlice []string
	for char := range vocabSet {
		vocabSlice = append(vocabSlice, char)
	}

	sort.Strings(vocabSlice)

	sortedVocabStr := strings.Join(vocabSlice, "")
	fmt.Println("Sorted Vocab: ", sortedVocabStr)

	stoi := mapCharToIdx(sortedVocabStr)
	encodedText := encodeString(text, stoi)

	fmt.Println("Tensor shape: ", len(encodedText))

	trainSize := int(0.9 * float64(len(encodedText)))
	train_data := encodedText[:trainSize]
	val_data := encodedText[trainSize:]

	fmt.Println("Train data size: ", len(train_data))
	fmt.Println("Val data size: ", len(val_data))

	block_size := 8
	batch_size := 4

	x, y := get_batch("train", train_data, val_data, batch_size, block_size)

	fmt.Println("Batch inputs (x) shape:", x.Shape())
	fmt.Println("Batch targets (y) shape:", y.Shape())
	fmt.Println("First batch inputs (x):", x)
	fmt.Println("First batch targets (y):", y)
	fmt.Println("-------------------")

	for b := 0; b < batch_size; b++ {
		for t := 0; t < block_size; t++ {
			// Slice and handle context correctly
			contextView, err := x.Slice(gorgonia.S(b), gorgonia.S(0, t+1))
			if err != nil {
				fmt.Println("Error slicing tensor x:", err)
				continue
			}
			contextData := contextView.Data()

			var context []int
			switch v := contextData.(type) {
			case []int:
				context = v
			case int:
				context = []int{v}
			default:
				fmt.Println("Unexpected type for context:", v)
				continue
			}

			targetView, err := y.At(b, t)
			if err != nil {
				fmt.Println("Error getting tensor y at index:", err)
				continue
			}
			target := targetView.(int)

			fmt.Printf("When input is %v the target: %d\n", context, target)
		}
	}
}
