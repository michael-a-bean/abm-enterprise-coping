
local({

  # the requested version of renv
  version <- "1.0.3"
  attr(version, "sha") <- NULL

  # the project directory
  project <- getwd()

  # use start-up diagnostics if enabled
  diagnostics <- Sys.getenv("RENV_STARTUP_DIAGNOSTICS", unset = "FALSE")
  if (diagnostics) {
    start <- Sys.time()
    profile <- FALSE
    on.exit({
      if (profile) {
        elapsed <- signif(difftime(Sys.time(), start, units = "auto"), 2L)
        message(sprintf("- renv activated [%s elapsed]", format(elapsed)))
      }
    }, add = TRUE)
  }

  # signal that we're loading renv during R startup
  Sys.setenv("RENV_R_INITIALIZING" = "true")
  on.exit(Sys.unsetenv("RENV_R_INITIALIZING"), add = TRUE)

  # signal that we've consented to use renv
  options(renv.consent = TRUE)

  # load the 'utils' package eagerly -- this ensures that renv shims, which
  # mask 'utils' packages, parsing code doesn't fail
  library(utils, quietly = TRUE)

  # check to see if renv has already been loaded
  if ("renv" %in% loadedNamespaces()) {

    # if renv has already been loaded, and it's the requested version of renv,
    # nothing to do
    spec <- .getNamespaceInfo(.getNamespace("renv"), "spec")
    if (identical(spec[["version"]], version))
      return(invisible(TRUE))

    # otherwise, unload and attempt to load the correct version of renv
    unloadNamespace("renv")

  }

  # load bootstrap tools
  `%||%` <- function(x, y) {
    if (is.null(x)) y else x
  }

  catf <- function(fmt, ..., appendLF = TRUE) {

    quiet <- getOption("renv.bootstrap.quiet", default = FALSE)
    if (quiet)
      return(invisible())

    msg <- sprintf(fmt, ...)
    cat(msg, file = stdout(), sep = if (appendLF) "\n" else "")

    invisible(msg)

  }

  header <- function(label,
                     ...,
                     prefix = "#",
                     suffix = "-",
                     n = min(getOption("width"), 78L))
  {
    label <- sprintf(label, ...)
    n <- max(n - nchar(label) - nchar(prefix) - 2L, 8L)
    if (n <= 0L)
      return(paste(prefix, label))

    tail <- paste(rep.int(suffix, n), collapse = "")
    paste0(prefix, " ", label, " ", tail)

  }

  startswith <- function(string, prefix) {
    substring(string, 1L, nchar(prefix)) == prefix
  }

  bootstrap <- function(version, library) {

    # attempt to download renv
    catf(header("Downloading renv %s", version))
    tarball <- tryCatch(renv_bootstrap_download(version), error = identity)
    if (inherits(tarball, "error"))
      stop("failed to download renv ", version)

    # now attempt to install
    catf(header("Installing renv %s", version))
    status <- tryCatch(renv_bootstrap_install(version, tarball, library), error = identity)
    if (inherits(status, "error"))
      stop("failed to install renv ", version)

  }

  renv_bootstrap_download <- function(version) {

    methods <- c(
      renv_bootstrap_download_cran_latest,
      renv_bootstrap_download_cran_archive,
      renv_bootstrap_download_github
    )

    for (method in methods) {
      path <- tryCatch(method(version), error = identity)
      if (is.character(path) && file.exists(path))
        return(path)
    }

    stop("all download methods failed")

  }

  renv_bootstrap_download_cran_latest <- function(version) {

    spec <- renv_bootstrap_download_cran_latest_find(version)
    type  <- spec$type
    repos <- spec$repos

    baseurl <- utils::contrib.url(repos = repos, type = type)
    ext <- if (identical(type, "source"))
      ".tar.gz"
    else if (Sys.info()[["sysname"]] == "Windows")
      ".zip"
    else
      ".tgz"
    name <- sprintf("renv_%s%s", version, ext)
    url <- paste(baseurl, name, sep = "/")

    destfile <- file.path(tempdir(), name)
    status <- tryCatch(
      renv_bootstrap_download_impl(url, destfile),
      error = identity
    )

    if (inherits(status, "error"))
      return(FALSE)

    destfile

  }

  renv_bootstrap_download_cran_latest_find <- function(version) {

    repos <- renv_bootstrap_repos()

    for (type in c("source", "binary")) {

      baseurl <- utils::contrib.url(repos = repos, type = type)
      db <- tryCatch(
        available.packages(contriburl = baseurl),
        error = identity,
        warning = identity
      )

      if (inherits(db, "condition"))
        next

      entry <- db[db[, "Package"] == "renv" & db[, "Version"] == version, ]
      if (nrow(entry) == 0L)
        next

      return(list(type = type, repos = repos))

    }

    stop("failed to find renv ", version, " in CRAN-like repositories")

  }

  renv_bootstrap_download_cran_archive <- function(version) {

    name <- sprintf("renv_%s.tar.gz", version)
    repos <- renv_bootstrap_repos()
    urls <- file.path(repos, "src/contrib/Archive/renv", name)

    destfile <- file.path(tempdir(), name)
    for (url in urls) {
      status <- tryCatch(
        renv_bootstrap_download_impl(url, destfile),
        error = identity
      )

      if (!inherits(status, "error"))
        return(destfile)
    }

    stop("failed to download renv ", version, " from CRAN-like repositories")

  }

  renv_bootstrap_download_github <- function(version) {

    enabled <- Sys.getenv("RENV_BOOTSTRAP_FROM_GITHUB", unset = "TRUE")
    if (!identical(enabled, "TRUE"))
      stop("GitHub bootstrap downloads are disabled")

    # prepare download options
    pat <- Sys.getenv("GITHUB_PAT")
    if (nzchar(Sys.which("curl")) && nzchar(pat)) {
      fmt <- "--location --fail --header \"Authorization: token %s\""
      extra <- sprintf(fmt, pat)
      saved <- options("download.file.method", "download.file.extra")
      options(download.file.method = "curl", download.file.extra = extra)
      on.exit(do.call(options, saved), add = TRUE)
    } else if (nzchar(Sys.which("wget")) && nzchar(pat)) {
      fmt <- "--header=\"Authorization: token %s\""
      extra <- sprintf(fmt, pat)
      saved <- options("download.file.method", "download.file.extra")
      options(download.file.method = "wget", download.file.extra = extra)
      on.exit(do.call(options, saved), add = TRUE)
    }

    url <- file.path("https://api.github.com/repos/rstudio/renv/tarball", version)
    name <- sprintf("renv_%s.tar.gz", version)
    destfile <- file.path(tempdir(), name)

    status <- tryCatch(
      renv_bootstrap_download_impl(url, destfile),
      error = identity
    )

    if (inherits(status, "error"))
      stop("failed to download renv from GitHub")

    destfile

  }

  renv_bootstrap_download_impl <- function(url, destfile) {

    mode <- "wb"

    # https://bugs.r-project.org/bugzilla/show_bug.cgi?id=17715
    fixup <-
      Sys.info()[["sysname"]] == "Windows" &&
      substring(url, 1L, 5L) == "file:"

    if (fixup)
      mode <- "w+b"

    args <- list(
      url      = url,
      destfile = destfile,
      mode     = mode,
      quiet    = TRUE
    )

    if ("headers" %in% names(formals(utils::download.file)))
      args$headers <- renv_bootstrap_download_custom_headers(url)

    do.call(utils::download.file, args)

  }

  renv_bootstrap_download_custom_headers <- function(url) {

    headers <- getOption("renv.download.headers")
    if (is.null(headers))
      return(character())

    if (!is.function(headers))
      stopf("'renv.download.headers' is not a function")

    headers(url)

  }

  renv_bootstrap_install <- function(version, tarball, library) {

    # attempt to install it into project library
    dir.create(library, showWarnings = FALSE, recursive = TRUE)
    output <- renv_bootstrap_install_impl(library, tarball)

    # check for successful install
    status <- attr(output, "status")
    if (is.null(status) || identical(status, 0L))
      return(TRUE)

    # an error occurred; report it
    header <- sprintf("Error installing renv %s", version)
    lines <- paste(rep.int("=", nchar(header)), collapse = "")
    text <- paste(c(header, lines, output), collapse = "\n")
    stop(text)

  }

  renv_bootstrap_install_impl <- function(library, tarball) {

    # invoke using system2
    bin <- R.home("bin")
    exe <- if (Sys.info()[["sysname"]] == "Windows") "R.exe" else "R"
    R <- file.path(bin, exe)

    args <- c(
      "--vanilla", "CMD", "INSTALL", "--no-multiarch",
      "-l", shQuote(path.expand(library)),
      shQuote(path.expand(tarball))
    )
    system2(R, args, stdout = TRUE, stderr = TRUE)

  }

  renv_bootstrap_repos <- function() {

    repos <- Sys.getenv("RENV_CONFIG_REPOS_OVERRIDE", unset = NA)
    if (!is.na(repos))
      return(repos)

    repos <- getOption("repos")
    if (length(repos) && is.character(repos))
      return(repos)

    c(CRAN = "https://cloud.r-project.org")

  }

  renv_bootstrap_library_root <- function(project) {

    prefix <- renv_bootstrap_library_root_prefix()

    # attempt to read from environment variable
    envvar <- Sys.getenv("RENV_PATHS_LIBRARY_ROOT", unset = NA)
    if (!is.na(envvar))
      return(envvar)

    # form path in project directory
    root <- file.path(project, prefix)

    root

  }

  renv_bootstrap_library_root_prefix <- function() {
    if (renv_bootstrap_standalone_mode())
      "renv/library"
    else
      "library"
  }

  renv_bootstrap_standalone_mode <- function() {
    TRUE
  }

  renv_bootstrap_library_path <- function(project) {

    root <- renv_bootstrap_library_root(project)

    path <- Sys.getenv("RENV_PATHS_LIBRARY", unset = NA)
    if (!is.na(path))
      return(path)

    prefix <- renv_bootstrap_platform_prefix()
    file.path(root, prefix)

  }

  renv_bootstrap_platform_prefix <- function() {
    paste(R.version$platform, R.version$major, R.version$minor, sep = "/")
  }

  renv_bootstrap_load <- function(project, libpath, version) {

    # try to load renv from the project library
    if (!requireNamespace("renv", lib.loc = libpath, quietly = TRUE))
      return(FALSE)

    # warn if the version of renv loaded does not match
    loadedversion <- utils::packageDescription("renv", fields = "Version")
    if (version != loadedversion) {

      # assume user wants to proceed if version mismatch detected
      fmt <- paste(
        "renv %1$s was loaded from project library, but renv %2$s is recorded in lockfile.",
        "Use `renv::record(\"renv@%1$s\")` to record this version in the lockfile.",
        "Use `renv::restore(packages = \"renv\")` to install renv %2$s into the project library.",
        sep = "\n"
      )
      catf(fmt, loadedversion, version)

    }

    TRUE

  }

  # load renv
  renv_bootstrap_run <- function(project, libpath, version) {

    # first, check if renv is installed, and matching requested version
    installed <- requireNamespace("renv", lib.loc = libpath, quietly = TRUE)
    if (installed) {

      # get the version of renv in the library
      installed_version <- utils::packageDescription("renv", lib.loc = libpath)$Version

      # if it matches the requested version, load and return
      if (identical(installed_version, version)) {
        return(renv_bootstrap_load(project, libpath, version))
      }

    }

    # failed to load from project library, try bootstrap
    bootstrap(version, libpath)

    # try again after bootstrap
    renv_bootstrap_load(project, libpath, version)

  }

  # construct path to library
  libpath <- renv_bootstrap_library_path(project)

  # run bootstrap code
  renv_bootstrap_run(project, libpath, version)

  # exit early if renv is not installed
  if (!"renv" %in% loadedNamespaces())
    return(invisible())

  # update profile
  profile <- renv:::renv_profile_get()
  renv:::renv_wd_set(project)

  on.exit({
    if (profile)
      renv:::renv_profile_set(profile)
  }, add = TRUE)

  # run user's R profile
  rprofile <- Sys.getenv("R_PROFILE_USER", unset = NA)
  if (!is.na(rprofile) && file.exists(rprofile)) {
    source(rprofile, local = TRUE)
  }

})
