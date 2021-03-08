# HTTP首部

| 类型     | 说明                                                 |
| -------- | ---------------------------------------------------- |
| 通用首部 | 是请求和响应都会用到的字段                           |
| 请求首部 | 是客户端向服务器发送请求时，报文中包含的首部字段     |
| 响应首部 | 是服务器向浏览器返回响应报文时，包含的首部字段       |
| 实体首部 | 是请求报文和响应报文中针对实体内容的首部字段         |
| 拓展首部 | 是非标准首部字段，由开发者根据自身需求自由定义和实现 |

## 通用首部

| 字段名            | 说明                                                   |
| ----------------- | ------------------------------------------------------ |
| Cache-Control     | 控制缓存行为                                           |
| Pragma            | HTTP/1.0遗留字段，也是用于控制缓存机制                 |
| Transfer-Encoding | 传输报文主体的编码方式                                 |
| Trailer           | 报文主体之后的首部字段，用于分块传输                   |
| Upgrade           | 检测HTTP协议是否可用更高版本                           |
| Connection        | 控制不再转发给代理的字段、连接的管理                   |
| Date              | 创建报文的日期                                         |
| Via               | 追踪客户端与服务器之间报文的传输路径，通常指代理服务器 |
| Warning           | 缓存相关警告                                           |

## 请求首部

| 字段名              | 说明                                            |
| ------------------- | ----------------------------------------------- |
| Accept              | 客户端可接受的媒体类型及相关优先级，q值表示权重 |
| Accept-Charset      | 客户端可接受的字符集及优先顺序                  |
| Accept-Encoding     | 客户端支持的内容编码及优先顺序                  |
| Accept-Language     | 客户端可处理的自然语言集，以及优先级            |
| Authorization       | 客户端的认证信息，一般是证书信息                |
| Host                | 请求资源所在服务器的主机名和端口号              |
| If-Match            | 比较实体标记                                    |
| If-Modified-Since   | 比较资源更新时间                                |
| If-None-Match       | 比较实体标记(与If-Match作用相反)                |
| If-Range            | 资源未更新时发送实体Byte的范围请求              |
| If-Unmodified-Since | 比较资源更新时间(与If-Modified-Since作用相反)   |
| Max-Forwards        | 最大传输逐跳数(TRACE或OPTIONS方法会用到)        |
| Range               | 范围请求的实体字节段                            |
| Referer             | 请求页面的原始url                               |
| TE                  | 传输编码及优先级                                |
| User-Agent          | 请求客户端的自身信息                            |

## 响应首部

| 字段名              | 说明                             |
| ------------------- | -------------------------------- |
| Accept-Ranges       | 服务器是否接受字节范围请求       |
| Age                 | 服务器响应创建经过的时间         |
| ETag                | 资源配置信息                     |
| Location            | 服务器告知客户端重定向url        |
| Proxy-Authorization | 代理服务器向客户端发起的认证信息 |
| Retry-After         | 服务器告知客户端再次请求的时间   |
| Server              | 服务器应用名、版本号等相关信息   |
| Vary                | 代理服务器的缓存管理信息         |
| WWW-Authorization   | 服务器对客户端的认证信息         |

## 实体首部

| 字段名           | 说明                   |
| ---------------- | ---------------------- |
| Allow            | 资源支持的请求方法     |
| Content-Encoding | 实体内容的编码方式     |
| Content-Language | 实体内容的自然语言集   |
| Content-Length   | 实体内容字节长度       |
| Content-Location | 实体内容替代url        |
| Content-MD5      | 实体内容的报文摘要     |
| Content-Range    | 实体内容的位置范围     |
| Content-Type     | 实体内容对应的媒体类型 |
| Expires          | 实体内容失效日期       |
| Last-Modified    | 实体内容最后修改日期   |

# 那个header和跨域有关

origin

> 参考

[HTTP——需要知道的协议 - SegmentFault 思否](https://segmentfault.com/a/1190000009537864)

[跨域资源共享 CORS 详解 - 阮一峰的网络日志 (ruanyifeng.com)](https://www.ruanyifeng.com/blog/2016/04/cors.html)