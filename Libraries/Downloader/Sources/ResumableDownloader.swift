import Foundation

#if canImport(UIKit)
  import UIKit
#endif

public final class ResumableDownloader: NSObject {
  public enum Backend {
    case automatic
    case downloadTask
  }

  public typealias ProgressHandler =
    (
      _ totalBytesWritten: Int64, _ totalBytesExpectedToWrite: Int64, _ isComplete: Bool,
      _ error: Error?
    ) -> Void

  public let remoteUrl: URL
  public let localUrl: URL
  public let sha256: String?
  public let backend: Backend
  /// Sent with every request (the HEAD probe, ranged GETs and the download task), for example
  /// an `Authorization` header for a server that needs a token. A redirect to another host drops
  /// `Authorization` (`RedirectHeaderPolicy`), so a token never reaches a CDN.
  public let headers: [String: String]

  private let lock = NSLock()
  private var probeSession: URLSession? = nil
  private var probeTask: URLSessionDataTask? = nil
  private var activeBackend: DownloadBackend? = nil
  private var generation: UInt64 = 0

  #if canImport(UIKit)
    private var backgroundTask: UIBackgroundTaskIdentifier? = nil
  #endif

  public init(
    remoteUrl: URL, localUrl: URL, sha256: String?, backend: Backend = .automatic,
    headers: [String: String] = [:]
  ) {
    self.remoteUrl = remoteUrl
    self.localUrl = localUrl
    self.sha256 = sha256
    self.backend = backend
    self.headers = headers
  }

  deinit {
    cancel()
    endBackgroundTask()
  }

  @discardableResult
  public static func probe(
    remoteUrl: URL, headers: [String: String] = [:],
    completionHandler: @escaping (Result<URLDownloadProbe, Error>) -> Void
  ) -> URLSessionDataTask {
    let probeSession = URLSession(configuration: .default)
    var request = URLRequest(url: remoteUrl, headers: headers)
    request.httpMethod = "HEAD"
    let probeTask = probeSession.dataTask(with: request) { _, response, error in
      defer {
        probeSession.finishTasksAndInvalidate()
      }
      if let error {
        completionHandler(.failure(DownloadError.local(error)))
        return
      }
      guard let httpResponse = response as? HTTPURLResponse else {
        completionHandler(.failure(DownloadError.unexpected))
        return
      }
      guard let probe = URLDownloadProbe(httpResponse: httpResponse) else {
        if (200..<300).contains(httpResponse.statusCode) {
          completionHandler(.failure(DownloadError.unexpected))
        } else {
          completionHandler(.failure(DownloadError.server(httpResponse.statusCode)))
        }
        return
      }
      completionHandler(.success(probe))
    }
    probeTask.delegate = RedirectHeaderPolicy()
    probeTask.resume()
    return probeTask
  }

  public func resume(_ handler: @escaping ProgressHandler) {
    let probeSession = URLSession(configuration: .default)

    lock.lock()
    cancelLocked()
    generation += 1
    let token = generation
    self.probeSession = probeSession
    beginBackgroundTaskLocked()
    lock.unlock()

    var request = URLRequest(url: remoteUrl, headers: headers)
    request.httpMethod = "HEAD"

    var probeTask: URLSessionDataTask!
    probeTask = probeSession.dataTask(with: request) { [weak self] _, response, error in
      self?.handleProbeResponse(
        token: token, response: response, error: error, userHandler: handler)
    }

    lock.lock()
    guard generation == token else {
      lock.unlock()
      probeTask.cancel()
      probeSession.invalidateAndCancel()
      return
    }
    self.probeTask = probeTask
    lock.unlock()
    // The probe session has no delegate of its own; the task's handles its redirects.
    probeTask.delegate = RedirectHeaderPolicy()
    probeTask.resume()
  }

  public func cancel() {
    lock.lock()
    generation += 1
    cancelLocked()
    lock.unlock()
    endBackgroundTask()
  }

  public enum DownloadError: Error, CustomStringConvertible {
    case local(Error)
    case server(Int)
    case fileMismatch
    case unexpected

    public var description: String {
      switch self {
      case .local(let error):
        return error.localizedDescription
      case .server(let statusCode):
        return "Server Error Code \(statusCode)"
      case .fileMismatch:
        return "File downloaded doesn't match the known good one"
      case .unexpected:
        return "Unexpected Error"
      }
    }
  }
}

extension ResumableDownloader {
  fileprivate func handleProbeResponse(
    token: UInt64, response: URLResponse?, error: Error?, userHandler: @escaping ProgressHandler
  ) {
    lock.lock()
    guard generation == token else {
      lock.unlock()
      return
    }
    probeTask = nil
    probeSession?.finishTasksAndInvalidate()
    probeSession = nil
    lock.unlock()

    if let nsError = error as? NSError,
      nsError.domain == NSURLErrorDomain,
      nsError.code == URLError.cancelled.rawValue
    {
      return
    }

    let probe = (response as? HTTPURLResponse).flatMap { URLDownloadProbe(httpResponse: $0) }

    let wrappedHandler: ProgressHandler = {
      [weak self] bytesWritten, totalBytes, isComplete, error in
      self?.handleBackendEvent(
        token: token,
        totalBytesWritten: bytesWritten,
        totalBytesExpectedToWrite: totalBytes,
        isComplete: isComplete,
        error: error,
        userHandler: userHandler)
    }

    if backend == .automatic,
      let probe,
      probe.supportsSegmentedDownloads
    {
      let backend = SegmentedResumableDownloaderBackend(
        remoteUrl: remoteUrl,
        localUrl: localUrl,
        sha256: sha256,
        probe: probe,
        headers: headers,
        handler: wrappedHandler)
      lock.lock()
      guard generation == token else {
        lock.unlock()
        backend.cancel()
        return
      }
      self.activeBackend = backend
      lock.unlock()
      backend.start()
      return
    }

    if let httpResponse = response as? HTTPURLResponse, !shouldFallbackToDownloadTask(httpResponse)
    {
      handleBackendEvent(
        token: token,
        totalBytesWritten: 0,
        totalBytesExpectedToWrite: 0,
        isComplete: false,
        error: DownloadError.server(httpResponse.statusCode),
        userHandler: userHandler)
      return
    }

    if let error {
      handleBackendEvent(
        token: token,
        totalBytesWritten: 0,
        totalBytesExpectedToWrite: 0,
        isComplete: false,
        error: DownloadError.local(error),
        userHandler: userHandler)
      return
    }

    let backend = URLSessionDownloadTaskResumableDownloaderBackend(
      remoteUrl: remoteUrl,
      localUrl: localUrl,
      sha256: sha256,
      expectedLength: probe?.expectedLength,
      headers: headers,
      handler: wrappedHandler)
    lock.lock()
    guard generation == token else {
      lock.unlock()
      backend.cancel()
      return
    }
    self.activeBackend = backend
    lock.unlock()
    backend.start()
  }

  fileprivate func handleBackendEvent(
    token: UInt64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64,
    isComplete: Bool,
    error: Error?,
    userHandler: @escaping ProgressHandler
  ) {
    lock.lock()
    guard generation == token else {
      lock.unlock()
      return
    }
    if isComplete || error != nil {
      activeBackend = nil
    }
    lock.unlock()

    userHandler(totalBytesWritten, totalBytesExpectedToWrite, isComplete, error)

    if isComplete || error != nil {
      endBackgroundTask()
    }
  }

  fileprivate func shouldFallbackToDownloadTask(_ response: HTTPURLResponse) -> Bool {
    switch response.statusCode {
    case 200..<300:
      return true
    case 400, 403, 405, 501:
      return true
    default:
      return false
    }
  }

  fileprivate func cancelLocked() {
    probeTask?.cancel()
    probeTask = nil
    probeSession?.invalidateAndCancel()
    probeSession = nil
    activeBackend?.cancel()
    activeBackend = nil
  }

  fileprivate func beginBackgroundTaskLocked() {
    #if canImport(UIKit)
      if backgroundTask == nil {
        backgroundTask = UIApplication.shared.beginBackgroundTask()
      }
    #endif
  }

  fileprivate func endBackgroundTask() {
    #if canImport(UIKit)
      guard let backgroundTask = backgroundTask else { return }
      UIApplication.shared.endBackgroundTask(backgroundTask)
      self.backgroundTask = nil
    #endif
  }
}

/// Drops `Authorization` from a redirected request that goes to another host. URLSession copies
/// a request's headers onto the request it redirects to, and a server that redirects a download
/// to a CDN (Hugging Face sends LFS files to signed CDN URLs) must not hand that host the token.
final class RedirectHeaderPolicy: NSObject, URLSessionTaskDelegate {
  static func redirectedRequest(_ request: URLRequest, from response: HTTPURLResponse)
    -> URLRequest
  {
    let fromHost = response.url?.host?.lowercased()
    let toHost = request.url?.host?.lowercased()
    guard fromHost != toHost || response.url?.scheme != request.url?.scheme else { return request }
    var request = request
    request.setValue(nil, forHTTPHeaderField: "Authorization")
    return request
  }

  func urlSession(
    _ session: URLSession, task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
    completionHandler: @escaping (URLRequest?) -> Void
  ) {
    completionHandler(Self.redirectedRequest(request, from: response))
  }
}

extension URLRequest {
  init(url: URL, headers: [String: String]) {
    self.init(url: url)
    for (field, value) in headers {
      setValue(value, forHTTPHeaderField: field)
    }
  }
}

protocol DownloadBackend: AnyObject {
  func start()
  func cancel()
}
