> 自己对context有啥理解，以及自己如何使用，content.Done()是干嘛的

在 Go http包的Server中，每一个请求在都有一个对应的 goroutine 去处理。请求处理函数通常会启动额外的 goroutine 用来访问后端服务，比如数据库和RPC服务。用来处理一个请求的 goroutine 通常需要访问一些与请求特定的数据，比如终端用户的身份认证信息、验证相关的token、请求的截止时间。 当一个请求被取消或超时时，所有用来处理该请求的 goroutine 都应该迅速退出，然后系统才能释放这些 goroutine 占用的资源。

# 如何阻塞线程

先说一下go里面如何使用管道来阻塞线程

```go
func Hello(ch chan int)  {
    fmt.Println("hello everybody , I'm asong")
    ch <- 1
}
func main()  {
    ch := make(chan int)
    go Hello(ch)
    <-ch
    fmt.Println("Golang梦工厂")
}
```

# 场景引入

go的context其实是用于控制并发的，首先我们有如下场景：

当一个gorouine启动后，一般情况下我们都是无法控制它的，只能等待它结束，如果有一个不断运行的goroutine的话，我们如何通知呢？一个比较优雅的方法就是使用chan+select，示例代码如下：

```go
func main() {
	stop := make(chan bool)
	go func() {
		for {
			select {
			case <-stop:
				fmt.Println("监控退出，停止了...")
				return
			default:
				fmt.Println("goroutine监控中...")
				time.Sleep(2 * time.Second)
			}
		}
	}()
	time.Sleep(10 * time.Second)
	fmt.Println("可以了，通知监控停止")
	stop <- true
	//为了检测监控过是否停止，如果没有监控输出，就表示停止了
	time.Sleep(5 * time.Second)
}
/**
输出结果
goroutine监控中...
goroutine监控中...
goroutine监控中...
goroutine监控中...
goroutine监控中...
可以了，通知监控停止
监控退出，停止了...
**/
```

例子中我们定义一个 stop 的 chan，通知它结束后台 goroutine。实现也非常简单，在后台 goroutine 中，使用 select 判断 stop 是否可以接收到值，如果可以接收到，就表示可以退出停止了；如果没有接收到，就会执行 default 里的监控逻辑，继续监控，只到收到 stop 的通知。

这样的方式好是好，但是如果有多个goroutine的话，那么应该怎么办呢，答案是使用context

# 初识context

上面说的这种场景是存在的，比如一个网络请求 Request，每个 Request 都需要开启一个 goroutine 做一些事情，这些 goroutine 又可能会开启其它的 goroutine。所以我们需要一种可以跟踪 goroutine 的方案，才可以达到控制它们的目的，这就是Go语言为我们提供的 Context，称之为上下文非常贴切，它就是 goroutine 的上下文。

我们使用context来重写一下

```go
func main() {
    // 使用context.Background()返回一个空的context，作为我们的context树的根节点
    // 然后我们使用context.WithCancel来创建一个可以取消的context
    // 第一个返回的是context对象，第二个返回的是一个回调函数，使用这个回调函数，我们可以取消context
	ctx, cancel := context.WithCancel(context.Background())
	go func(ctx context.Context) {
		for {
			select {
                // 这个ctx.done() 就是在监听context，判断是否结束了
			case <-ctx.Done():
				fmt.Println("监控退出，停止了...")
				return
			default:
				fmt.Println("goroutine监控中...")
				time.Sleep(2 * time.Second)
			}
		}
	}(ctx)

	time.Sleep(10 * time.Second)
	fmt.Println("可以了，通知监控停止")
	// 调用cancel函数，我们就可以发出取消的指令，这样我们的goroutine就会就会收到信号，结束函数
	cancel()
	
	//为了检测监控过是否停止，如果没有监控输出，就表示停止了
	time.Sleep(5 * time.Second)
}
```

如果有多个goroutine的情况下，我们应该怎么做呢？

```go
func main() {
	ctx, cancel := context.WithCancel(context.Background())
	go watch(ctx, "【监控1】")
	go watch(ctx, "【监控2】")
	go watch(ctx, "【监控3】")
	time.Sleep(10 * time.Second)
	fmt.Println("可以了，通知监控停止")
	cancel()
	//为了检测监控过是否停止，如果没有监控输出，就表示停止了
	time.Sleep(5 * time.Second)
}
func watch(ctx context.Context, name string) {
	for {
		select {
		case <-ctx.Done():
			fmt.Println(name, "监控退出，停止了...")
			return
		default:
			fmt.Println(name, "goroutine监控中...")
			time.Sleep(2 * time.Second)
		}
	}
}

/**
【监控3】 goroutine监控中...
【监控1】 goroutine监控中...
【监控2】 goroutine监控中...
【监控1】 goroutine监控中...
【监控2】 goroutine监控中...
【监控3】 goroutine监控中...
【监控1】 goroutine监控中...
【监控2】 goroutine监控中...
【监控3】 goroutine监控中...
【监控1】 goroutine监控中...
【监控2】 goroutine监控中...
【监控3】 goroutine监控中...
【监控3】 goroutine监控中...
【监控1】 goroutine监控中...
【监控2】 goroutine监控中...
可以了，通知监控停止
【监控2】 监控退出，停止了...
【监控3】 监控退出，停止了...
【监控1】 监控退出，停止了...
**/
```

示例中启动了 3 个监控 goroutine 进行不断的监控，每一个都使用了 Context 进行跟踪，当我们使用 cancel 函数通知取消时，这 3 个 goroutine 都会被结束。这就是 Context 的控制能力，它就像一个控制器一样，按下开关后，所有基于这个 Context 或者衍生的子 Context 都会收到通知，这时就可以进行清理操作了，最终释放 goroutine，这就优雅的解决了 goroutine 启动后不可控的问题。

# context接口方法

```go
type Context interface {
	Deadline() (deadline time.Time, ok bool)

	Done() <-chan struct{}

	Err() error

	Value(key interface{}) interface{}
}
```

可以看到context的接口还是比较简洁的就只有这几个方法，下面来一个一个介绍

## 方法介绍

### Deadline

Deadline 方法是获取设置的截止时间的意思，第一个返回式是截止时间，到了这个时间点，Context 会自动发起取消请求；第二个返回值 ok==false 时表示没有设置截止时间，如果需要取消的话，需要调用取消函数进行取消。

### Done方法

Done 方法返回一个只读的 chan，类型为 struct{}，我们在 goroutine 中，如果该方法返回的 chan 可以读取，则意味着 parent context 已经发起了取消请求，我们通过 Done 方法收到这个信号后，就应该做清理操作，然后退出 goroutine，释放资源。

### Err方法

Err 方法返回取消的错误原因，因为什么 Context 被取消。

### Value方法

Value 方法获取该 Context 上绑定的值，是一个键值对，所以要通过一个 Key 才可以获取对应的值，这个值一般是线程安全的。

## 常用方法Done介绍

如果 Context 取消的时候，我们就可以得到一个关闭的 chan，关闭的 chan 是可以读取的，所以只要可以读取的时候，就意味着收到 Context 取消的信号了，以下是这个方法的经典用法。

```go
 func Stream(ctx context.Context, out chan<- Value) error {
  	for {
  		v, err := DoSomething(ctx)
  		if err != nil {
  			return err
  		}
  		select {
        // 判断context是否完成，如果取消了，那我们就返回错误信息
  		case <-ctx.Done():
  			return ctx.Err()
  		case out <- v:
  		}
  	}
  }
```

## context默认提供的方法

Context 接口并不需要我们实现，Go 内置已经帮我们实现了 2 个，我们代码中最开始都是以这两个内置的作为最顶层的 partent context，衍生出更多的子 Context。

```go
var (
	background = new(emptyCtx)
	todo       = new(emptyCtx)
)

func Background() Context {
	return background
}

func TODO() Context {
	return todo
}
```

一个是 Background，主要用于 main 函数、初始化以及测试代码中，作为 Context 这个树结构的最顶层的 Context，也就是根 Context。

一个是 TODO，它目前还不知道具体的使用场景，如果我们不知道该使用什么 Context 的时候，可以使用这个。

它们两个本质上都是 emptyCtx 结构体类型，是一个不可取消，没有设置截止时间，没有携带任何值的 Context。

## emptyCtx的源码

```go
type emptyCtx int

func (*emptyCtx) Deadline() (deadline time.Time, ok bool) {
	return
}

func (*emptyCtx) Done() <-chan struct{} {
	return nil
}

func (*emptyCtx) Err() error {
	return nil
}

func (*emptyCtx) Value(key interface{}) interface{} {
	return nil
}
```

这就是 emptyCtx 实现 Context 接口的方法，可以看到，这些方法什么都没做，返回的都是 nil 或者零值。

# context的继承

上面我们使用context.Background返回了一个空的context，那么我们如何产生子context呢，我们可以使用context包提供的with函数

```go
func WithCancel(parent Context) (ctx Context, cancel CancelFunc)
func WithDeadline(parent Context, deadline time.Time) (Context, CancelFunc)
func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc)
func WithValue(parent Context, key, val interface{}) Context
```

这四个 With 函数，接收的都有一个 partent 参数，就是父 Context，我们要基于这个父 Context 创建出子 Context 的意思，这种方式可以理解为子 Context 对父 Context 的继承，也可以理解为基于父 Context 的衍生。

**通过这些函数，就创建了一颗 Context 树，树的每个节点都可以有任意多个子节点，节点层级可以有任意多个。**

WithCancel 函数，传递一个父 Context 作为参数，返回子 Context，以及一个取消函数用来取消 Context。 WithDeadline 函数，和 WithCancel 差不多，它会多传递一个截止时间参数，意味着到了这个时间点，会自动取消 Context，当然我们也可以不等到这个时候，可以提前通过取消函数进行取消。

WithTimeout 和 WithDeadline 基本上一样，这个表示是超时自动取消，是多少时间后自动取消 Context 的意思。

WithValue 函数和取消 Context 无关，它是为了生成一个绑定了一个键值对数据的 Context，这个绑定的数据可以通过 Context.Value 方法访问到，后面我们会专门讲。

大家可能留意到，前三个函数都返回一个取消函数 CancelFunc，这是一个函数类型，它的定义非常简单。

```go
type CancelFunc func()
```

这就是取消函数的类型，该函数可以**取消一个 Context，以及这个节点 Context下所有的所有的 Context，**不管有多少层级。

# 使用context来传递数据

```go
var key string = "name"
func main() {
	ctx, cancel := context.WithCancel(context.Background())
	//附加值
	valueCtx := context.WithValue(ctx, key, "【监控1】")
	go watch(valueCtx)
	time.Sleep(10 * time.Second)
	fmt.Println("可以了，通知监控停止")
	cancel()
	//为了检测监控过是否停止，如果没有监控输出，就表示停止了
	time.Sleep(5 * time.Second)
}
func watch(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
		   //取出值
			fmt.Println(ctx.Value(key), "监控退出，停止了...")
			return
		default:
		//取出值
			fmt.Println(ctx.Value(key), "goroutine监控中...")
			time.Sleep(2 * time.Second)
		}
	}
}
```

记住，使用 WithValue 传值，一般是必须的值，不要什么值都传递。

## Context 使用原则

- 不要把 Context 放在结构体中，要以参数的方式传递
- 以 Context 作为参数的函数方法，应该把 Context 作为第一个参数，放在第一位。
- 给一个函数方法传递 Context 的时候，不要传递 nil，如果不知道传递什么，就使用 context.TODO
- Context 的 Value 相关方法应该传递必须的数据，不要什么数据都使用这个传递
- Context 是线程安全的，可以放心的在多个 goroutine 中传递





> 参考

[Go Context - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/58967892)

