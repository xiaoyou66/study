REST指Representational State Transfer，可以翻译为“表现层状态转化”

#### 主要思想

- 对网络上的所有资源，都有一个**统一资源标识符** URI(Uniform Resource Identifier)；
- 这些资源可以有多种表现形式，即REST中的“表现层”Representation，比如，文本可以用txt格式表现，也可以用HTML格式、XML格式、JSON格式表现。URI只代表资源的实体，不代表它的形式；
- “无状态(Stateless)”思想：服务端不应该保存客户端状态，只需要处理当前的请求，不需了解请求的历史，客户端每一次请求中包含处理该请求所需的一切信息；
- 客户端使用HTTP协议中的 GET/POST/PUT/DELETE 方法对服务器的资源进行操作，即REST中的”状态转化“

#### 设计原则

- URL设计
  - 最好只使用名词，而使用 GET/POST/PUT/DELETE 方法的不同表示不同的操作；比如使用`POST /user`代替`/user/create`
  - GET：获取资源；POST：新建/更新资源；PUT：更新资源；DELETE：删除资源；
  - 对于只支持GET/POST的客户端，使用`X-HTTP-Method-Override`属性，覆盖POST方法；
  - 避免多级URL，比如使用`GET /authors/12?categories=2`代替`GET /authors/12/categories/2`；
  - 避免在URI中带上版本号。不同的版本，可以理解成同一种资源的不同表现形式，所以应该采用同一个URI，版本号可以在HTTP请求头信息的Accept字段中进行区分
- 状态码：服务器应该返回尽可能精确的状态码，客户端只需查看状态码，就可以判断出发生了什么情况。见计算机网络部分 -- [HTTP请求有哪些常见状态码？](https://github.com/wolverinn/Waking-Up/blob/master/Computer Network.md#HTTP请求有哪些常见状态码)
- 服务器回应：在响应中放上其它API的链接，方便用户寻找