plugins {
    java
    application
}

group = "io.drift"
version = "0.1.0"

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.11.3"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

application {
    mainClass = "io.drift.Main"
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "2g"
    testLogging {
        events("passed", "skipped", "failed")
    }
}

// Dumps every representable value and every product of every value pair, for the
// Python side to check against exact rational arithmetic. See python/verify_oracle.py.
tasks.register<JavaExec>("dumpTables") {
    group = "verification"
    mainClass = "io.drift.tools.DumpTables"
    classpath = sourceSets["main"].runtimeClasspath
}

tasks.register<JavaExec>("train") {
    group = "application"
    mainClass = "io.drift.train.TrainMain"
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs = listOf("-Xmx2g")
}
