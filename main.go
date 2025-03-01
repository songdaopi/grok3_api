package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// GrokClient 定义了与 Grok3 API 交互的客户端
type GrokClient struct {
	newUrl         string
	deleteUrl      string
	headers        map[string]string
	isReasoning    bool
	keepChat       bool
	ignoreThinking bool
}

// NewGrokClient 创建一个新的 GrokClient 实例
func NewGrokClient(cookies string, isReasoning bool, keepChat bool, ignoreThinking bool) *GrokClient {
	return &GrokClient{
		newUrl:    "https://grok.com/rest/app-chat/conversations/new",
		deleteUrl: "https://grok.com/rest/app-chat/conversations/%s",
		headers: map[string]string{
			"accept":             "*/*",
			"accept-language":    "en-GB,en;q=0.9",
			"content-type":       "application/json",
			"origin":             "https://grok.com",
			"priority":           "u=1, i",
			"referer":            "https://grok.com/",
			"sec-ch-ua":          `"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"`,
			"sec-ch-ua-mobile":   "?0",
			"sec-ch-ua-platform": `"macOS"`,
			"sec-fetch-dest":     "empty",
			"sec-fetch-mode":     "cors",
			"sec-fetch-site":     "same-origin",
			"sec-gpc":            "1",
			"user-agent":         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
			"cookie":             cookies,
		},
		isReasoning:    isReasoning,
		keepChat:       keepChat,
		ignoreThinking: ignoreThinking,
	}
}

// preparePayload 准备发送到 Grok3 API 的请求体
func (c *GrokClient) preparePayload(message string, isReasoning bool) map[string]any {
	return map[string]any{
		"customInstructions":        "",
		"deepsearchPreset":          "",
		"disableSearch":             false,
		"enableImageGeneration":     true,
		"enableImageStreaming":      true,
		"enableSideBySide":          true,
		"fileAttachments":           []string{},
		"forceConcise":              false,
		"imageAttachments":          []string{},
		"imageGenerationCount":      2,
		"isPreset":                  false,
		"isReasoning":               isReasoning,
		"message":                   message,
		"modelName":                 "grok-3",
		"returnImageBytes":          false,
		"returnRawGrokInXaiRequest": false,
		"sendFinalMetadata":         true,
		"temporary":                 false,
		"toolOverrides":             map[string]any{},
	}
}

// getModelName returns the model name.
func (c *GrokClient) getModelName() string {
	if c.isReasoning {
		return grok3ReasoningModelName
	} else {
		return grok3ModelName
	}
}

type RequestBody struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Stream           bool   `json:"stream"`
	Grok3Cookies     any    `json:"grok3Cookies,omitempty"`
	CookieIndex      uint   `json:"cookieIndex,omitempty"` // start from 1, 0 is auto(selecting cookies in turn)
	TextBeforePrompt string `json:"textBeforePrompt,omitempty"`
	TextAfterPrompt  string `json:"textAfterPrompt,omitempty"`
	KeepChat         int    `json:"keepChat,omitempty"`       // > 0 is true, == 0 is false
	IgnoreThinking   int    `json:"ignoreThinking,omitempty"` // > 0 is true, == 0 is false
}

type ResponseToken struct {
	Result struct {
		Response struct {
			Token      string `json:"token"`
			IsThinking bool   `json:"isThinking"`
		} `json:"response"`
	} `json:"result"`
}

type ResponseConversationId struct {
	Result struct {
		Conversation struct {
			ConversationId string `json:"conversationId"`
		} `json:"conversation"`
	} `json:"result"`
}

type ModelData struct {
	Id       string `json:"id"`
	Object   string `json:"object"`
	Owned_by string `json:"owned_by"`
}

type ModelList struct {
	Object string      `json:"object"`
	Data   []ModelData `json:"data"`
}

const grok3ModelName = "grok-3"
const grok3ReasoningModelName = "grok-3-reasoning"

const completionsPath = "/v1/chat/completions"
const listModelsPath = "/v1/models"

var grok3Token *string
var grok3Cookies []string
var keepChat *bool
var textBeforePrompt *string
var textAfterPrompt *string
var ignoreThinking *bool
var httpProxy *string
var httpClient = &http.Client{}
var nextCookieIndex = struct {
	sync.Mutex
	index uint // start from 0
}{}

// sendMessage 发送消息到 Grok3 API 并返回响应
func (c *GrokClient) sendMessage(message string, stream bool) (io.ReadCloser, error) {
	payload := c.preparePayload(message, c.isReasoning)
	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	req, err := http.NewRequest(http.MethodPost, c.newUrl, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	for key, value := range c.headers {
		req.Header.Set(key, value)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("the Grok API error: %d %s", resp.StatusCode, resp.Status)
	}

	if stream {
		return resp.Body, nil
	} else {
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %v", err)
		}
		return io.NopCloser(bytes.NewReader(body)), nil
	}
}

// deleteConversation 删除整个对话
func (c *GrokClient) deleteConversation(conversationId string) error {
	req, err := http.NewRequest(http.MethodDelete, fmt.Sprintf(c.deleteUrl, conversationId), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %v", err)
	}

	for key, value := range c.headers {
		req.Header.Set(key, value)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("the Grok API error: %d %s", resp.StatusCode, resp.Status)
	}
	defer resp.Body.Close()
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response body: %v", err)
	}

	return nil
}

// OpenAIChatCompletionChunk 定义了 OpenAI 兼容的流式响应块
type OpenAIChatCompletionChunk struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// OpenAIChatCompletion 定义了 OpenAI 兼容的完整响应
type OpenAIChatCompletion struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// createOpenAIStreamingResponse 处理流式响应并转换为 OpenAI 格式
func (c *GrokClient) createOpenAIStreamingResponse(grokStream io.Reader) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			log.Println("Streaming unsupported")
			http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
			return
		}

		completionID := "chatcmpl-" + uuid.New().String()

		// 发送起始块
		startChunk := OpenAIChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   c.getModelName(),
			Choices: []struct {
				Index int `json:"index"`
				Delta struct {
					Role    string `json:"role,omitempty"`
					Content string `json:"content,omitempty"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Index: 0,
					Delta: struct {
						Role    string `json:"role,omitempty"`
						Content string `json:"content,omitempty"`
					}{
						Role: "assistant",
					},
					FinishReason: "",
				},
			},
		}
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(startChunk))
		flusher.Flush()

		// 处理流式数据
		isThinking := false
		buffer := make([]byte, 1024)
		var conversationId string
		for {
			n, err := grokStream.Read(buffer)
			if err != nil {
				if err == io.EOF {
					break
				}
				log.Printf("Error reading stream: %v", err)
				return
			}

			chunk := string(buffer[:n])
			lines := strings.SplitSeq(chunk, "\n")
			for line := range lines {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}

				if !c.keepChat && conversationId == "" {
					var respConversation ResponseConversationId
					if err := json.Unmarshal([]byte(line), &respConversation); err == nil {
						conversationId = respConversation.Result.Conversation.ConversationId
					}
				}

				var token ResponseToken
				if err := json.Unmarshal([]byte(line), &token); err != nil {
					continue
				}

				respToken := token.Result.Response.Token
				if c.ignoreThinking && token.Result.Response.IsThinking {
					continue
				} else if token.Result.Response.IsThinking {
					if !isThinking {
						respToken = "<think>\n" + respToken
					}
					isThinking = true
				} else if isThinking {
					respToken = respToken + "\n</think>\n\n"
					isThinking = false
				}

				if respToken != "" {
					chunk := OpenAIChatCompletionChunk{
						ID:      completionID,
						Object:  "chat.completion.chunk",
						Created: time.Now().Unix(),
						Model:   c.getModelName(),
						Choices: []struct {
							Index int `json:"index"`
							Delta struct {
								Role    string `json:"role,omitempty"`
								Content string `json:"content,omitempty"`
							} `json:"delta"`
							FinishReason string `json:"finish_reason"`
						}{
							{
								Index: 0,
								Delta: struct {
									Role    string `json:"role,omitempty"`
									Content string `json:"content,omitempty"`
								}{
									Content: respToken,
								},
								FinishReason: "",
							},
						},
					}
					fmt.Fprintf(w, "data: %s\n\n", mustMarshal(chunk))
					flusher.Flush()
				}
			}
		}

		// 发送结束块
		finalChunk := OpenAIChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   c.getModelName(),
			Choices: []struct {
				Index int `json:"index"`
				Delta struct {
					Role    string `json:"role,omitempty"`
					Content string `json:"content,omitempty"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			}{
				{
					Index: 0,
					Delta: struct {
						Role    string `json:"role,omitempty"`
						Content string `json:"content,omitempty"`
					}{},
					FinishReason: "stop",
				},
			},
		}
		fmt.Fprintf(w, "data: %s\n\n", mustMarshal(finalChunk))
		flusher.Flush()

		// 结束流
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()

		// 删除对话
		if !c.keepChat && conversationId != "" {
			if err := c.deleteConversation(conversationId); err != nil {
				log.Printf("Deleting conversation error: %v", err)
				http.Error(w, fmt.Sprintf("Deleting conversation error: %v", err), http.StatusInternalServerError)
				return
			}
		}
	}
}

// createOpenAIFullResponse 处理包含全部内容的响应并转换为 OpenAI 格式
func (c *GrokClient) createOpenAIFullResponse(grokFull io.Reader) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var fullResponse strings.Builder
		buf := new(strings.Builder)
		_, err := io.Copy(buf, grokFull)
		if err != nil {
			log.Printf("Reading response error: %v", err)
			http.Error(w, fmt.Sprintf("Reading response error: %v", err), http.StatusInternalServerError)
			return
		}

		var conversationId string
		isThinking := false
		lines := strings.SplitSeq(buf.String(), "\n")
		for line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			if !c.keepChat && conversationId == "" {
				var respConversation ResponseConversationId
				if err := json.Unmarshal([]byte(line), &respConversation); err == nil {
					conversationId = respConversation.Result.Conversation.ConversationId
				}
			}

			var token ResponseToken
			if err := json.Unmarshal([]byte(line), &token); err != nil {
				continue
			}

			respToken := token.Result.Response.Token
			if c.ignoreThinking && token.Result.Response.IsThinking {
				continue
			} else if token.Result.Response.IsThinking {
				if !isThinking {
					respToken = "<think>\n" + respToken
				}
				isThinking = true
			} else if isThinking {
				respToken = respToken + "\n</think>\n\n"
				isThinking = false
			}

			fullResponse.WriteString(respToken)
		}

		openAIResponse := c.createOpenAIFullResponseBody(fullResponse.String())
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(openAIResponse); err != nil {
			log.Printf("Encoding response error: %v", err)
			http.Error(w, fmt.Sprintf("Encoding response error: %v", err), http.StatusInternalServerError)
			return
		}

		if !c.keepChat && conversationId != "" {
			if err := c.deleteConversation(conversationId); err != nil {
				log.Printf("Deleting conversation error: %v", err)
				http.Error(w, fmt.Sprintf("Deleting conversation error: %v", err), http.StatusInternalServerError)
				return
			}
		}
	}
}

// createOpenAIFullResponseBody 创建 OpenAI 兼容的完整响应
func (c *GrokClient) createOpenAIFullResponseBody(content string) OpenAIChatCompletion {
	return OpenAIChatCompletion{
		ID:      "chatcmpl-" + uuid.New().String(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   c.getModelName(),
		Choices: []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}{
			{
				Index: 0,
				Message: struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				}{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: "stop",
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     -1,
			CompletionTokens: -1,
			TotalTokens:      -1,
		},
	}
}

// mustMarshal 将对象序列化为 JSON 字符串
func mustMarshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// getCookieIndex 获取 cookie 的 index（从 1 算起）, len 要大于 0
func getCookieIndex(len int, cookieIndex uint) uint {
	if cookieIndex == 0 || cookieIndex > uint(len) {
		nextCookieIndex.Lock()
		defer nextCookieIndex.Unlock()
		index := nextCookieIndex.index
		nextCookieIndex.index = (nextCookieIndex.index + 1) % uint(len)

		return index % uint(len)
	} else {
		return cookieIndex - 1
	}
}

// handleRequest handles HTTP requests.
func handleRequest(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != completionsPath {
		log.Printf("Requested Path %s Not Found", r.URL.Path)
		http.Error(w, "Requested Path Not Found", http.StatusNotFound)
		return
	}

	if r.Method != http.MethodPost {
		log.Printf("Method %s Not Allowed", r.Method)
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	// 验证身份
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
		log.Println("Unauthorized: Bearer token required")
		http.Error(w, "Unauthorized: Bearer token required", http.StatusUnauthorized)
		return
	}

	token := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer "))
	if token != *grok3Token {
		log.Println("Unauthorized: Invalid token")
		http.Error(w, "Unauthorized: Invalid token", http.StatusUnauthorized)
		return
	}

	// 解析请求体
	body := RequestBody{KeepChat: -1, IgnoreThinking: -1}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		log.Println("Bad Request: Invalid JSON")
		http.Error(w, "Bad Request: Invalid JSON", http.StatusBadRequest)
		return
	}

	var cookie string
	if body.Grok3Cookies != nil {
		if ck, ok := body.Grok3Cookies.(string); ok {
			cookie = ck
		} else if list, ok := body.Grok3Cookies.([]any); ok {
			if len(list) > 0 {
				if ck, ok := list[getCookieIndex(len(list), body.CookieIndex)].(string); ok {
					cookie = ck
				}
			}
		}
	}
	cookie = strings.TrimSpace(cookie)
	if cookie == "" && len(grok3Cookies) > 0 {
		cookie = grok3Cookies[getCookieIndex(len(grok3Cookies), body.CookieIndex)]
	}
	cookie = strings.TrimSpace(cookie)
	if cookie == "" {
		log.Println("Error: No Grok3 cookie")
		http.Error(w, "Error: No Grok3 cookie", http.StatusBadRequest)
		return
	}

	messages := body.Messages
	if len(messages) == 0 {
		log.Println("Bad Request: No messages provided")
		http.Error(w, "Bad Request: No messages provided", http.StatusBadRequest)
		return
	}

	// 提取消息
	msg, err := json.Marshal(messages)
	if err != nil {
		log.Println("Bad Request: No messages provided")
		http.Error(w, "Bad Request: No messages provided", http.StatusBadRequest)
		return
	}
	if len(msg) <= 2 {
		log.Println("Bad Request: No messages provided")
		http.Error(w, "Bad Request: No user message found", http.StatusBadRequest)
		return
	}

	var beforePromptText string
	var afterPromptText string
	if body.TextBeforePrompt != "" {
		beforePromptText = body.TextBeforePrompt
	} else {
		beforePromptText = *textBeforePrompt
	}
	if body.TextAfterPrompt != "" {
		afterPromptText = body.TextAfterPrompt
	} else {
		afterPromptText = *textAfterPrompt
	}
	message := beforePromptText + string(msg) + afterPromptText

	isReasoning := false
	if strings.TrimSpace(body.Model) == grok3ReasoningModelName {
		isReasoning = true
	}

	keepConversation := false
	if body.KeepChat > 0 {
		keepConversation = true
	} else if body.KeepChat < 0 {
		keepConversation = *keepChat
	}

	ignoreThink := false
	if body.IgnoreThinking > 0 {
		ignoreThink = true
	} else if body.IgnoreThinking < 0 {
		ignoreThink = *ignoreThinking
	}

	// 初始化 GrokClient
	grokClient := NewGrokClient(cookie, isReasoning, keepConversation, ignoreThink)
	respReader, err := grokClient.sendMessage(message, body.Stream)
	if err != nil {
		log.Printf("Error: %v", err)
		http.Error(w, fmt.Sprintf("Error: %v", err), http.StatusInternalServerError)
		return
	}
	defer respReader.Close()

	if body.Stream {
		// 处理流式响应
		grokClient.createOpenAIStreamingResponse(respReader)(w, r)
	} else {
		// 处理完整响应
		grokClient.createOpenAIFullResponse(respReader)(w, r)
	}
}

// listModels lists all models.
func listModels(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != listModelsPath {
		log.Printf("Requested Path %s Not Found", r.URL.Path)
		http.Error(w, "Requested Path Not Found", http.StatusNotFound)
		return
	}

	if r.Method != http.MethodGet {
		log.Printf("Method %s Not Allowed", r.Method)
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	list := ModelList{
		Object: "list",
		Data: []ModelData{
			{
				Id:       grok3ModelName,
				Object:   "model",
				Owned_by: "xAI",
			},
			{
				Id:       grok3ReasoningModelName,
				Object:   "model",
				Owned_by: "xAI",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(list); err != nil {
		log.Printf("Encoding response error: %v", err)
		http.Error(w, fmt.Sprintf("Encoding response error: %v", err), http.StatusInternalServerError)
		return
	}
}

// main 启动 HTTP 服务器
func main() {
	grok3Token = flag.String("token", "", "Authentication token (GROK3_AUTH_TOKEN)")
	cookie := flag.String("cookie", "", "Grok3 cookie (GROK3_COOKIE)")
	keepChat = flag.Bool("keepChat", false, "Don't delete the chat after request")
	textBeforePrompt = flag.String("textBeforePrompt", "For the data below, entries with the role 'system' are system information, entries with the role 'assistant' are messages you have previously sent, entries with the role 'user' are messages sent by the user. You need to respond to the user's last message accordingly based on the corresponding data.", "Text before the prompt")
	textAfterPrompt = flag.String("textAfterPrompt", "", "Text after the prompt")
	ignoreThinking = flag.Bool("ignoreThinking", false, "Ignore the thinking content while using the reasoning model")
	httpProxy = flag.String("httpProxy", "", "HTTP/SOCKS5 proxy")
	port := flag.Uint("port", 8180, "Server port")
	flag.Parse()

	if *port > 65535 {
		log.Fatalf("Server port %d is greater than 65535", *port)
	}

	*grok3Token = strings.TrimSpace(*grok3Token)
	if *grok3Token == "" {
		*grok3Token = os.Getenv("GROK3_AUTH_TOKEN")
		if *grok3Token == "" {
			log.Fatal("未设置 Authentication token (GROK3_AUTH_TOKEN)")
		}
	}

	*cookie = strings.TrimSpace(*cookie)
	if *cookie == "" {
		*cookie = strings.TrimSpace(os.Getenv("GROK3_COOKIE"))
	}
	if *cookie != "" {
		err := json.Unmarshal([]byte(*cookie), &grok3Cookies)
		if err != nil {
			grok3Cookies = []string{*cookie}
		}
	}

	*httpProxy = strings.TrimSpace(*httpProxy)
	if *httpProxy != "" {
		proxyURL, err := url.Parse(*httpProxy)
		if err == nil {
			httpClient.Transport = &http.Transport{
				Proxy: http.ProxyURL(proxyURL),
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				ForceAttemptHTTP2:   true,
				MaxIdleConns:        10,
				IdleConnTimeout:     120 * time.Second,
				TLSHandshakeTimeout: 20 * time.Second,
			}
		} else {
			log.Fatalf("Parsing HTTP/SOCKS5 proxy error：%v", err)
		}
	}

	http.HandleFunc(completionsPath, handleRequest)
	http.HandleFunc(listModelsPath, listModels)
	log.Printf("Server starting on :%d", *port)

	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
}
