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
    mainClass = "io.drift.train.TrainMain"
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

// Inner products under every policy, 200 seeds, three lengths. Writes results/dot_sweep.csv.
tasks.register<JavaExec>("dotSweep") {
    group = "application"
    mainClass = "io.drift.tools.DotSweep"
    classpath = sourceSets["main"].runtimeClasspath
}

// The dot-product sweep on the wide model's real first-layer tensors. Appends to results/real_dot.csv.
tasks.register<JavaExec>("realDotSweep") {
    group = "application"
    mainClass = "io.drift.tools.RealDotSweep"
    classpath = sourceSets["main"].runtimeClasspath
}

// The training sweep. Pass flags with --args, e.g. --args="--seeds 1,2,3 --steps 2000".
tasks.register<JavaExec>("train") {
    group = "application"
    mainClass = "io.drift.train.TrainMain"
    classpath = sourceSets["main"].runtimeClasspath
    jvmArgs = listOf("-Xmx3g")
}
